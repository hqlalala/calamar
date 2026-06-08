"""Tool registry — auto-discovery and unified dispatch."""

from __future__ import annotations

from typing import Any

from calamar.tools.base import Tool, ToolResult, ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.spec.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def to_openai_tools(self) -> list[dict[str, Any]]:
        return [t.spec.to_openai_schema() for t in self._tools.values()]

    async def dispatch(self, name: str, args: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(error=f"Unknown tool: {name}")
        return await tool.execute(**args)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
