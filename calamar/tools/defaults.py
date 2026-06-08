"""Default tool registry — registers all built-in tools."""

from __future__ import annotations

from calamar.tools.file import FileEditTool, FileReadTool, FileWriteTool
from calamar.tools.registry import ToolRegistry
from calamar.tools.search import ListDirectoryTool, SearchCodeTool
from calamar.tools.terminal import TerminalTool


def create_default_registry(working_dir: str = ".") -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(TerminalTool(working_dir=working_dir))
    registry.register(FileReadTool())
    registry.register(FileWriteTool())
    registry.register(FileEditTool())
    registry.register(SearchCodeTool(working_dir=working_dir))
    registry.register(ListDirectoryTool(working_dir=working_dir))
    return registry
