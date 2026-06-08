"""File tools — read, write, and edit files."""

from __future__ import annotations

import os
from dataclasses import dataclass

from calamar.tools.base import ToolParameter, ToolResult, ToolSpec

MAX_READ = 50_000


@dataclass
class FileReadTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="file_read",
            description=(
                "Read the contents of a file. Returns the full file with "
                "line numbers. For large files, use offset and limit."
            ),
            parameters=[
                ToolParameter(
                    name="path", type="string",
                    description="Path to the file to read",
                ),
                ToolParameter(
                    name="offset", type="integer",
                    description="Start from this line number (0-based)",
                    required=False, default=0,
                ),
                ToolParameter(
                    name="limit", type="integer",
                    description="Max number of lines to read (default: all)",
                    required=False, default=0,
                ),
            ],
        )

    async def execute(
        self, path: str = "", offset: int = 0, limit: int = 0, **_: object,
    ) -> ToolResult:
        if not path:
            return ToolResult(error="No path provided")

        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return ToolResult(error=f"File not found: {path}")
        if os.path.isdir(path):
            return ToolResult(error=f"Path is a directory: {path}. Use terminal with 'ls' instead.")

        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError as e:
            return ToolResult(error=f"Cannot read file: {e}")

        total = len(lines)
        if offset > 0:
            lines = lines[offset:]
        if limit > 0:
            lines = lines[:limit]

        numbered = []
        for i, line in enumerate(lines, start=max(offset, 0) + 1):
            numbered.append(f"{i:>5}\t{line.rstrip()}")

        content = "\n".join(numbered)
        if len(content) > MAX_READ:
            content = content[:MAX_READ] + "\n...[truncated]"

        return ToolResult(
            output=content,
            metadata={"total_lines": total, "path": path},
        )


@dataclass
class FileWriteTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="file_write",
            description=(
                "Create a new file or completely overwrite an existing file. "
                "Prefer file_edit for modifying existing files."
            ),
            parameters=[
                ToolParameter(
                    name="path", type="string",
                    description="Path to the file to write",
                ),
                ToolParameter(
                    name="content", type="string",
                    description="Content to write to the file",
                ),
            ],
        )

    async def execute(
        self, path: str = "", content: str = "", **_: object,
    ) -> ToolResult:
        if not path:
            return ToolResult(error="No path provided")

        path = os.path.expanduser(path)
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as e:
            return ToolResult(error=f"Cannot write file: {e}")

        lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return ToolResult(
            output=f"Wrote {lines} lines to {path}",
            metadata={"path": path, "lines": lines},
        )


@dataclass
class FileEditTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="file_edit",
            description=(
                "Edit a file by replacing an exact string with a new string. "
                "The old_string must match exactly (including whitespace). "
                "Use this instead of file_write for modifying existing files."
            ),
            parameters=[
                ToolParameter(
                    name="path", type="string",
                    description="Path to the file to edit",
                ),
                ToolParameter(
                    name="old_string", type="string",
                    description="Exact string to find and replace",
                ),
                ToolParameter(
                    name="new_string", type="string",
                    description="String to replace with",
                ),
            ],
        )

    async def execute(
        self,
        path: str = "",
        old_string: str = "",
        new_string: str = "",
        **_: object,
    ) -> ToolResult:
        if not path:
            return ToolResult(error="No path provided")
        if not old_string:
            return ToolResult(error="No old_string provided")

        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return ToolResult(error=f"File not found: {path}")

        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
        except OSError as e:
            return ToolResult(error=f"Cannot read file: {e}")

        count = content.count(old_string)
        if count == 0:
            return ToolResult(error="old_string not found in file")
        if count > 1:
            return ToolResult(
                error=f"old_string found {count} times. Provide more context to make it unique.",
            )

        new_content = content.replace(old_string, new_string, 1)

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
        except OSError as e:
            return ToolResult(error=f"Cannot write file: {e}")

        return ToolResult(
            output=f"Edited {path}: replaced 1 occurrence",
            metadata={"path": path},
        )
