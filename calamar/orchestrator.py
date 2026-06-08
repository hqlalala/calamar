"""Orchestrator — multi-agent coordination with automatic complexity upgrade."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum

from calamar.context import ContextBuilder, user_message
from calamar.events import Event, RoleChangeEvent, TextEvent
from calamar.loop import AgentLoop
from calamar.roles import DEFAULT, EXECUTOR, PLANNER, VERIFIER, AgentRole


class TaskComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class OrchestratorConfig:
    max_retries: int = 2
    auto_upgrade: bool = True
    force_plan: bool = False


COMPLEX_KEYWORDS = {
    "refactor", "重构", "redesign", "migrate", "迁移",
    "rewrite", "重写", "overhaul", "architecture", "架构",
}

MODERATE_KEYWORDS = {
    "fix", "修复", "bug", "add", "添加", "implement", "实现",
    "update", "更新", "change", "改", "modify", "修改",
    "create", "创建", "build", "构建",
}


def classify_complexity(user_input: str) -> TaskComplexity:
    """Classify task complexity from user input."""
    lower = user_input.lower()

    for kw in COMPLEX_KEYWORDS:
        if kw in lower:
            return TaskComplexity.COMPLEX

    file_refs = re.findall(r"[\w/]+\.\w{1,5}", user_input)
    if len(file_refs) >= 3:
        return TaskComplexity.COMPLEX

    for kw in MODERATE_KEYWORDS:
        if kw in lower:
            return TaskComplexity.MODERATE

    return TaskComplexity.SIMPLE


class Orchestrator:
    """Manages multi-agent collaboration over a shared AgentLoop.

    For simple tasks, runs the default single-agent mode.
    For complex tasks, orchestrates Planner → Executor → Verifier
    with automatic retry on verification failure.
    """

    def __init__(
        self,
        agent: AgentLoop,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self._agent = agent
        self._config = config or OrchestratorConfig()

    @property
    def agent(self) -> AgentLoop:
        return self._agent

    async def run(self, user_input: str) -> AsyncIterator[Event]:
        complexity = classify_complexity(user_input)

        if self._config.force_plan:
            complexity = TaskComplexity.COMPLEX

        if not self._config.auto_upgrade or complexity == TaskComplexity.SIMPLE:
            async for event in self._agent.run(user_input):
                yield event
            return

        if complexity == TaskComplexity.MODERATE:
            async for event in self._run_with_verification(user_input):
                yield event
            return

        async for event in self._run_tri_agent(user_input):
            yield event

    async def _run_tri_agent(self, user_input: str) -> AsyncIterator[Event]:
        """Full Planner → Executor → Verifier pipeline."""

        # Phase 1: Planning
        yield RoleChangeEvent(
            from_role="default", to_role="planner",
        )
        self._switch_role(PLANNER)
        plan_prompt = (
            f"Analyze this request and create a step-by-step plan. "
            f"Identify the relevant files and the changes needed. "
            f"Do NOT make any edits.\n\n"
            f"Request: {user_input}"
        )

        plan_text = ""
        async for event in self._agent.run(plan_prompt):
            yield event
            if isinstance(event, TextEvent):
                plan_text += event.text

        # Phase 2: Execution
        yield RoleChangeEvent(
            from_role="planner", to_role="executor",
        )
        self._switch_role(EXECUTOR)
        exec_prompt = (
            f"Execute the following plan. Make the minimum necessary edits.\n\n"
            f"Plan:\n{plan_text}\n\n"
            f"Original request: {user_input}"
        )

        async for event in self._agent.run(exec_prompt):
            yield event

        # Phase 3: Verification with retry
        for attempt in range(self._config.max_retries + 1):
            yield RoleChangeEvent(
                from_role="executor", to_role="verifier",
            )
            self._switch_role(VERIFIER)
            verify_prompt = (
                f"Verify that the changes correctly address the request. "
                f"Run tests if available. Check for bugs or missing edge cases.\n\n"
                f"Original request: {user_input}"
            )

            verify_text = ""
            async for event in self._agent.run(verify_prompt):
                yield event
                if isinstance(event, TextEvent):
                    verify_text += event.text

            if self._is_pass(verify_text):
                break

            if attempt < self._config.max_retries:
                yield RoleChangeEvent(
                    from_role="verifier", to_role="executor",
                )
                self._switch_role(EXECUTOR)
                fix_prompt = (
                    f"The verifier found issues. Fix them:\n\n"
                    f"{verify_text}"
                )
                async for event in self._agent.run(fix_prompt):
                    yield event

        # Return to default
        self._switch_role(DEFAULT)
        yield RoleChangeEvent(
            from_role="verifier", to_role="default",
        )

    async def _run_with_verification(
        self, user_input: str,
    ) -> AsyncIterator[Event]:
        """Execute + Verify (skip planning for moderate tasks)."""
        async for event in self._agent.run(user_input):
            yield event

        yield RoleChangeEvent(
            from_role="default", to_role="verifier",
        )
        self._switch_role(VERIFIER)
        verify_prompt = (
            f"Verify the changes just made. Run tests if available. "
            f"Report pass/fail.\n\nOriginal request: {user_input}"
        )

        verify_text = ""
        async for event in self._agent.run(verify_prompt):
            yield event
            if isinstance(event, TextEvent):
                verify_text += event.text

        if not self._is_pass(verify_text):
            yield RoleChangeEvent(
                from_role="verifier", to_role="default",
            )
            self._switch_role(DEFAULT)
            fix_prompt = f"The verifier found issues. Fix them:\n\n{verify_text}"
            async for event in self._agent.run(fix_prompt):
                yield event
        else:
            self._switch_role(DEFAULT)
            yield RoleChangeEvent(
                from_role="verifier", to_role="default",
            )

    def _switch_role(self, role: AgentRole) -> None:
        self._agent._role = role
        self._agent._context_builder = ContextBuilder(
            system_prompt=role.system_prompt,
            tool_schemas=self._agent._tools.to_openai_tools(),
        )

    @staticmethod
    def _is_pass(verify_text: str) -> bool:
        lower = verify_text.lower()
        fail_signals = [
            "fail", "error", "bug", "issue", "problem", "incorrect",
            "missing", "broken", "wrong", "not working",
            "失败", "错误", "问题",
        ]
        pass_signals = [
            "pass", "correct", "looks good", "lgtm", "verified",
            "all tests pass", "no issues", "working correctly",
            "通过", "正确", "没有问题",
        ]

        fail_count = sum(1 for s in fail_signals if s in lower)
        pass_count = sum(1 for s in pass_signals if s in lower)

        return pass_count >= fail_count
