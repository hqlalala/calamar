"""Agent roles — swappable system prompt + toolset profiles."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentRole:
    name: str
    system_prompt: str
    tool_names: list[str] = field(default_factory=list)
    model_override: str | None = None


DEFAULT = AgentRole(
    name="default",
    system_prompt=(
        "You are Calamar, an AI coding agent. You help users with software "
        "engineering tasks: fixing bugs, adding features, refactoring code, "
        "and explaining codebases. Be concise and precise."
    ),
)

PLANNER = AgentRole(
    name="planner",
    system_prompt=(
        "You are the planning agent. Your job is to understand the user's "
        "request, locate relevant code, and produce a step-by-step plan. "
        "Do NOT make edits — only analyze and plan."
    ),
    tool_names=["read_file", "search_code", "list_directory", "web_search"],
)

EXECUTOR = AgentRole(
    name="executor",
    system_prompt=(
        "You are the execution agent. Follow the plan precisely. Make the "
        "minimum necessary edits. Prefer editing over creating new files."
    ),
    tool_names=["read_file", "edit_file", "write_file", "terminal"],
)

VERIFIER = AgentRole(
    name="verifier",
    system_prompt=(
        "You are the verification agent. Check whether the edits correctly "
        "solve the task. Run tests if available. Report pass/fail with "
        "specific reasons."
    ),
    tool_names=["read_file", "terminal", "search_code"],
)

BUILTIN_ROLES = {
    "default": DEFAULT,
    "planner": PLANNER,
    "executor": EXECUTOR,
    "verifier": VERIFIER,
}
