"""AgentLoop — the core turn-based agent loop."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator

from calamar.config import Config
from calamar.context import (
    ContextBuilder,
    ContextCompactor,
    Message,
    assistant_message,
    tool_result_message,
    user_message,
)
from calamar.events import (
    CompactionEvent,
    ErrorEvent,
    Event,
    TextEvent,
    ToolEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from calamar.git.workflow import GitWorkflow
from calamar.middleware import (
    CostMiddleware,
    InputGuardrail,
    MiddlewarePipeline,
    OutputGuardrail,
    TimingMiddleware,
)
from calamar.providers import create_provider
from calamar.roles import DEFAULT, AgentRole
from calamar.router import ModelRouter
from calamar.tools.defaults import create_default_registry
from calamar.tools.registry import ToolRegistry


class AgentLoop:
    """Single-threaded, turn-based agent loop.

    Design principles (learned from Claude Code, Codex, Aider, SWE-Bench):
    - Flat message history, no threading
    - Static prefix + dynamic suffix to maximize prompt cache hits
    - Steering queue for mid-execution user instructions
    - Progressive context compaction
    """

    def __init__(
        self,
        config: Config,
        tools: ToolRegistry | None = None,
        role: AgentRole | None = None,
    ) -> None:
        self._config = config
        self._role = role or DEFAULT
        self._tools = tools or create_default_registry(
            working_dir=config.project_root or ".",
        )
        self._history: list[Message] = []
        self._steering_queue: asyncio.Queue[str] = asyncio.Queue()
        self._router = ModelRouter(config)
        self._compactor = ContextCompactor(
            config.context_window, config.compaction.threshold,
        )

        self._provider = create_provider(config)

        self._cost = CostMiddleware(config.budget.daily_limit_usd)
        self._pipeline = (
            MiddlewarePipeline()
            .use(InputGuardrail())
            .use(self._cost)
            .use(TimingMiddleware())
            .use(OutputGuardrail())
        )

        self._context_builder = ContextBuilder(
            system_prompt=self._role.system_prompt or config.system_prompt,
            tool_schemas=self._tools.to_openai_tools(),
        )

        self._inject_repo_map(config.project_root or ".")

        self._git = GitWorkflow(config.project_root or ".")

    @property
    def history(self) -> list[Message]:
        return self._history

    @property
    def role(self) -> AgentRole:
        return self._role

    @property
    def git(self) -> GitWorkflow:
        return self._git

    @property
    def provider(self):
        return self._provider

    def set_role(self, role: AgentRole) -> None:
        """Switch the agent's active role (system prompt + tool profile)."""
        self._role = role
        self._context_builder = ContextBuilder(
            system_prompt=role.system_prompt or self._config.system_prompt,
            tool_schemas=self._tools.to_openai_tools(),
        )

    def clear_history(self) -> None:
        """Clear all conversation history."""
        self._history.clear()

    def inject_steering(self, instruction: str) -> None:
        self._steering_queue.put_nowait(instruction)

    _FILE_TOOLS = {"file_write", "file_edit", "terminal"}

    async def run(self, user_input: str) -> AsyncIterator[Event]:
        turn_id = uuid.uuid4().hex[:12]
        yield TurnStartEvent(turn_id=turn_id)

        self._history.append(user_message(user_input))

        tool_call_count = 0
        total_tokens = 0
        total_cost = 0.0
        has_file_changes = False

        for _iteration in range(self._config.max_turns):
            self._drain_steering()

            model = self._select_model(user_input)
            messages = self._context_builder.build(self._history)
            tools_schema = self._context_builder.tool_schemas or None

            try:
                accumulated_text = ""
                tool_calls = []
                finish_reason = "stop"

                async for delta in self._provider.stream(
                    model=model,
                    messages=messages,
                    tools=tools_schema,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                ):
                    if delta.text:
                        accumulated_text += delta.text
                        yield TextEvent(text=delta.text, streaming=True)
                    if delta.tool_calls:
                        tool_calls = delta.tool_calls
                    if delta.finish_reason:
                        finish_reason = delta.finish_reason
                    if delta.usage:
                        total_tokens += delta.usage.total_tokens
                        total_cost += delta.usage.cost_usd
            except Exception as e:
                yield ErrorEvent(error=str(e), recoverable=False)
                break

            if finish_reason == "stop" or not tool_calls:
                self._history.append(assistant_message(accumulated_text))
                break

            tool_calls_raw = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in tool_calls
            ]

            self._history.append(Message(
                role="assistant",
                content=accumulated_text,
                tool_calls=tool_calls_raw,
            ))

            for tc in tool_calls:
                name = tc.name
                args = tc.arguments

                ctx = await self._pipeline.execute(
                    tool_name=name,
                    tool_args=args,
                    tool_call_id=tc.id,
                    executor=self._tools.dispatch,
                )

                result_text = ""
                if ctx.result:
                    result_text = ctx.result.error or ctx.result.output

                self._history.append(
                    tool_result_message(tc.id, result_text)
                )

                yield ToolEvent(
                    tool_name=name,
                    tool_args=args,
                    result=result_text,
                    duration_ms=ctx.metadata.get("duration_ms", 0),
                )
                tool_call_count += 1
                if name in self._FILE_TOOLS:
                    has_file_changes = True

            if self._compactor.needs_compaction(self._history):
                before = len(self._history)
                self._history = await self._compactor.compact(self._history)
                yield CompactionEvent(from_messages=before, to_messages=len(self._history))

        if has_file_changes and await self._git.is_repo():
            summary = self._summarize_turn(user_input, tool_call_count)
            await self._git.auto_commit(summary)

        yield TurnEndEvent(
            turn_id=turn_id,
            tool_calls=tool_call_count,
            tokens_used=total_tokens,
            cost_usd=total_cost,
        )

    def _select_model(self, user_input: str) -> str:
        if self._role.model_override:
            return self._role.model_override
        return self._router.route(user_input, has_code_context=bool(self._history))

    def _drain_steering(self) -> None:
        while not self._steering_queue.empty():
            try:
                instruction = self._steering_queue.get_nowait()
                self._history.append(
                    Message(role="user", content=f"[steering] {instruction}")
                )
            except asyncio.QueueEmpty:
                break

    def _inject_repo_map(self, project_root: str) -> None:
        try:
            from calamar.code_index.repo_map import RepoMap
        except ImportError:
            return

        try:
            repo_map = RepoMap(project_root)
            text = repo_map.render(max_tokens=2000)
            if text.strip():
                self._context_builder.add_context_file(
                    f"# Repository Structure\n\n{text}"
                )
        except Exception:
            # Repo map is best-effort; failures here must not block agent startup.
            pass

    def _summarize_turn(self, user_input: str, tool_calls: int) -> str:
        truncated = user_input[:80].replace("\n", " ").strip()
        if len(user_input) > 80:
            truncated += "..."
        return f"{truncated} ({tool_calls} tool calls)"
