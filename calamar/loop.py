"""AgentLoop — the core turn-based agent loop."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

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
    TokenUsage,
    ToolEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from calamar.middleware import (
    CostMiddleware,
    InputGuardrail,
    MiddlewarePipeline,
    OutputGuardrail,
    TimingMiddleware,
)
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

        self._client = AsyncOpenAI(
            api_key=config.api_key or "not-set",
            base_url=config.base_url,
        )

        self._cost = CostMiddleware(config.budget.daily_limit_usd)
        self._pipeline = (
            MiddlewarePipeline()
            .use(InputGuardrail())
            .use(TimingMiddleware())
            .use(OutputGuardrail())
        )

        self._context_builder = ContextBuilder(
            system_prompt=self._role.system_prompt or config.system_prompt,
            tool_schemas=self._tools.to_openai_tools(),
        )

        self._inject_repo_map(config.project_root or ".")

    @property
    def history(self) -> list[Message]:
        return self._history

    @property
    def role(self) -> AgentRole:
        return self._role

    def inject_steering(self, instruction: str) -> None:
        self._steering_queue.put_nowait(instruction)

    async def run(self, user_input: str) -> AsyncIterator[Event]:
        turn_id = uuid.uuid4().hex[:12]
        yield TurnStartEvent(turn_id=turn_id)

        self._history.append(user_message(user_input))

        tool_call_count = 0
        total_tokens = 0
        total_cost = 0.0

        for _iteration in range(self._config.max_turns):
            self._drain_steering()

            model = self._select_model(user_input)
            messages = self._context_builder.build(self._history)
            tools_schema = self._context_builder.tool_schemas or None

            try:
                response = await self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools_schema,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                )
            except Exception as e:
                yield ErrorEvent(error=str(e), recoverable=False)
                break

            choice = response.choices[0]
            usage = self._record_usage(response, model)
            total_tokens += usage.total_tokens
            total_cost += usage.cost_usd

            if choice.finish_reason == "stop" or not choice.message.tool_calls:
                text = choice.message.content or ""
                self._history.append(assistant_message(text))
                if text:
                    yield TextEvent(text=text)
                break

            tool_calls_raw = []
            for tc in choice.message.tool_calls:
                tool_calls_raw.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

            self._history.append(Message(
                role="assistant",
                content=choice.message.content or "",
                tool_calls=tool_calls_raw,
            ))

            for tc in choice.message.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

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

            if self._compactor.needs_compaction(self._history):
                before = len(self._history)
                self._history = await self._compactor.compact(self._history)
                yield CompactionEvent(from_tokens=before, to_tokens=len(self._history))

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

    def _record_usage(self, response: Any, model: str) -> TokenUsage:
        usage = response.usage
        if usage is None:
            return TokenUsage(model=model)

        token_usage = TokenUsage(
            prompt_tokens=usage.prompt_tokens or 0,
            completion_tokens=usage.completion_tokens or 0,
            total_tokens=usage.total_tokens or 0,
            model=model,
        )
        cache = getattr(usage, "prompt_tokens_details", None)
        if cache:
            token_usage.cache_read_tokens = getattr(cache, "cached_tokens", 0)
        return token_usage

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
            pass
