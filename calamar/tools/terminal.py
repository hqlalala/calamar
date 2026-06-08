"""Terminal tool — execute shell commands."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from calamar.tools.base import ToolParameter, ToolResult, ToolSpec

MAX_OUTPUT = 10_000


@dataclass
class TerminalTool:
    working_dir: str = "."
    default_timeout: int = 30

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="terminal",
            description=(
                "Execute a shell command and return its output. "
                "Use for running tests, installing packages, checking "
                "git status, listing files, or any CLI operation."
            ),
            parameters=[
                ToolParameter(
                    name="command",
                    type="string",
                    description="The shell command to execute",
                ),
                ToolParameter(
                    name="timeout",
                    type="integer",
                    description="Timeout in seconds (default 30)",
                    required=False,
                    default=30,
                ),
            ],
        )

    async def execute(self, command: str = "", timeout: int = 0, **_: object) -> ToolResult:
        if not command:
            return ToolResult(error="No command provided")

        timeout = timeout or self.default_timeout

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            return ToolResult(error=f"Command timed out after {timeout}s")
        except OSError as e:
            return ToolResult(error=f"Failed to execute: {e}")

        out = stdout.decode(errors="replace")
        err = stderr.decode(errors="replace")

        if len(out) > MAX_OUTPUT:
            out = out[:MAX_OUTPUT] + f"\n...[truncated, {len(out)} chars total]"

        output_parts = []
        if out.strip():
            output_parts.append(out.strip())
        if err.strip():
            output_parts.append(f"STDERR:\n{err.strip()}")

        result_text = "\n".join(output_parts) if output_parts else "(no output)"

        if proc.returncode != 0:
            return ToolResult(
                output=result_text,
                error=f"Exit code {proc.returncode}",
                metadata={"exit_code": proc.returncode},
            )

        return ToolResult(output=result_text, metadata={"exit_code": 0})
