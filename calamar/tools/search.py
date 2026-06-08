"""Search tools — grep code and find files."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from calamar.tools.base import ToolParameter, ToolResult, ToolSpec

MAX_RESULTS = 100


@dataclass
class SearchCodeTool:
    working_dir: str = "."

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="search_code",
            description=(
                "Search for a pattern in code files using grep. "
                "Returns matching lines with file paths and line numbers."
            ),
            parameters=[
                ToolParameter(
                    name="pattern", type="string",
                    description="Search pattern (regex supported)",
                ),
                ToolParameter(
                    name="path", type="string",
                    description="Directory or file to search in (default: current dir)",
                    required=False, default=".",
                ),
                ToolParameter(
                    name="include", type="string",
                    description="File glob pattern, e.g. '*.py' (default: all files)",
                    required=False,
                ),
            ],
        )

    async def execute(
        self,
        pattern: str = "",
        path: str = ".",
        include: str = "",
        **_: object,
    ) -> ToolResult:
        if not pattern:
            return ToolResult(error="No search pattern provided")

        cmd_parts = ["grep", "-rn", "--color=never"]
        if include:
            cmd_parts.extend(["--include", include])
        cmd_parts.extend(["--", pattern, path])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=15,
            )
        except asyncio.TimeoutError:
            proc.kill()
            return ToolResult(error="Search timed out")
        except OSError as e:
            return ToolResult(error=f"Search failed: {e}")

        output = stdout.decode(errors="replace").strip()
        if not output:
            return ToolResult(output="No matches found")

        lines = output.split("\n")
        if len(lines) > MAX_RESULTS:
            output = "\n".join(lines[:MAX_RESULTS])
            output += f"\n...[{len(lines)} total matches, showing first {MAX_RESULTS}]"

        return ToolResult(
            output=output,
            metadata={"match_count": min(len(lines), MAX_RESULTS)},
        )


@dataclass
class ListDirectoryTool:
    working_dir: str = "."

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="list_directory",
            description=(
                "List files and directories at a given path. "
                "Shows file sizes and types."
            ),
            parameters=[
                ToolParameter(
                    name="path", type="string",
                    description="Directory path to list (default: current dir)",
                    required=False, default=".",
                ),
                ToolParameter(
                    name="pattern", type="string",
                    description="Glob pattern to filter, e.g. '*.py'",
                    required=False,
                ),
            ],
        )

    async def execute(
        self, path: str = ".", pattern: str = "", **_: object,
    ) -> ToolResult:
        if pattern:
            cmd = f"find {path} -name '{pattern}' -maxdepth 3 | head -100"
        else:
            cmd = f"ls -la {path}"

        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=10,
            )
        except asyncio.TimeoutError:
            proc.kill()
            return ToolResult(error="Listing timed out")
        except OSError as e:
            return ToolResult(error=f"Listing failed: {e}")

        output = stdout.decode(errors="replace").strip()
        return ToolResult(output=output or "(empty directory)")
