"""Terminal renderer — rich-based event display."""

from __future__ import annotations

import sys

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from calamar.events import (
    CompactionEvent,
    ErrorEvent,
    Event,
    RoleChangeEvent,
    TextEvent,
    ToolEvent,
    TurnEndEvent,
    TurnStartEvent,
)


class TerminalRenderer:
    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose
        self._console = Console()
        self._streaming = False
        self._stream_buffer = ""
        self._live: Live | None = None
        self._thinking = False

    def render(self, event: Event) -> None:
        if self._thinking and not isinstance(event, TurnStartEvent):
            self._stop_thinking()

        if not isinstance(event, TextEvent) and self._streaming:
            self._finalize_stream()

        match event:
            case TextEvent(text=text, streaming=streaming):
                if streaming:
                    self._stream_chunk(text)
                else:
                    self._render_text(text)
            case ToolEvent():
                self._render_tool(event)
            case TurnStartEvent():
                self._start_thinking()
            case TurnEndEvent():
                self._render_turn_end(event)
            case CompactionEvent():
                if self._verbose:
                    self._console.print(
                        "[dim]context compacted[/dim]"
                    )
            case RoleChangeEvent(to_role=to_role):
                self._render_role_change(to_role)
            case ErrorEvent(error=error):
                self._console.print(f"[bold red]Error:[/bold red] {error}")

    def flush(self) -> None:
        """Flush any pending stream buffer (e.g., after interrupt)."""
        if self._thinking:
            self._stop_thinking()
        if self._streaming:
            self._finalize_stream()

    def _start_thinking(self) -> None:
        self._thinking = True
        self._console.print("[dim]Thinking...[/dim]", end="")
        sys.stdout.flush()

    def _stop_thinking(self) -> None:
        if not self._thinking:
            return
        self._thinking = False
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def _stream_chunk(self, text: str) -> None:
        """Append a streaming chunk and update the live markdown display."""
        self._stream_buffer += text
        if not self._streaming:
            self._streaming = True
            self._live = Live(
                Markdown(self._stream_buffer),
                console=self._console,
                refresh_per_second=15,
            )
            self._live.start()
        else:
            self._live.update(Markdown(self._stream_buffer))

    def _finalize_stream(self) -> None:
        """Stop the live display — rendered markdown stays on screen."""
        if self._live is not None:
            self._live.stop()
            self._live = None
        self._streaming = False
        self._stream_buffer = ""

    def _render_text(self, text: str) -> None:
        try:
            md = Markdown(text)
            self._console.print(md)
        except Exception:
            self._console.print(text)

    def _render_tool(self, event: ToolEvent) -> None:
        header = Text()
        header.append("  ", style="bold yellow")
        header.append(event.tool_name, style="bold yellow")

        args_display = ""
        if event.tool_name == "terminal":
            args_display = event.tool_args.get("command", "")
        elif event.tool_name in ("file_read", "file_write", "file_edit"):
            args_display = event.tool_args.get("path", "")
        elif event.tool_name == "search_code":
            args_display = event.tool_args.get("pattern", "")
        elif event.tool_name == "list_directory":
            args_display = event.tool_args.get("path", ".")
        else:
            args_display = str(event.tool_args)[:80]

        if args_display:
            header.append(f" {args_display}", style="dim")

        if event.duration_ms > 0:
            header.append(f" ({event.duration_ms}ms)", style="dim")

        self._console.print(header)

        result_str = str(event.result) if event.result else ""

        if event.tool_name == "file_edit" and result_str:
            self._render_edit_diff(event)
        elif event.tool_name == "terminal" and result_str:
            self._render_terminal_output(result_str)
        elif result_str and self._verbose:
            truncated = result_str[:500]
            if len(result_str) > 500:
                truncated += "..."
            self._console.print(
                Panel(truncated, border_style="dim", expand=False)
            )

    def _render_edit_diff(self, event: ToolEvent) -> None:
        old = event.tool_args.get("old_string", "")
        new = event.tool_args.get("new_string", "")
        if not old and not new:
            return

        diff = Text()
        for line in old.splitlines():
            diff.append(f"  - {line}\n", style="red")
        for line in new.splitlines():
            diff.append(f"  + {line}\n", style="green")

        self._console.print(diff, end="")

    def _render_terminal_output(self, result_str: str) -> None:
        lines = result_str.splitlines()
        max_lines = 10 if self._verbose else 3
        shown = lines[:max_lines]
        remaining = len(lines) - max_lines

        if shown:
            output = Text()
            for line in shown:
                output.append(f"    {line}\n", style="dim")
            if remaining > 0:
                output.append(f"    ... {remaining} more lines\n", style="dim italic")
            self._console.print(output, end="")

    def _render_turn_end(self, event: TurnEndEvent) -> None:
        parts = []
        if event.tool_calls > 0:
            parts.append(f"{event.tool_calls} tool calls")
        if event.tokens_used > 0:
            parts.append(f"{event.tokens_used} tokens")
        if event.cache_read_tokens > 0:
            pct = event.cache_read_tokens * 100 // max(event.tokens_used, 1)
            parts.append(f"cache {pct}%")
        if event.cost_usd > 0:
            parts.append(f"${event.cost_usd:.4f}")

        if parts:
            summary = " | ".join(parts)
            self._console.print(f"\n[dim]{summary}[/dim]")

    _ROLE_LABELS = {
        "planner": "Planning",
        "executor": "Executing",
        "verifier": "Verifying",
        "default": "Done",
    }

    def _render_role_change(self, to_role: str) -> None:
        label = self._ROLE_LABELS.get(to_role, to_role)
        if to_role == "default":
            self._console.print(f"\n[dim]── {label} ──[/dim]")
        else:
            self._console.print(f"\n[bold cyan]── {label} ──[/bold cyan]")
