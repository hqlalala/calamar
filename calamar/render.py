"""Terminal renderer — rich-based event display."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from calamar.events import (
    CompactionEvent,
    ErrorEvent,
    Event,
    TextEvent,
    ToolEvent,
    TurnEndEvent,
    TurnStartEvent,
)

console = Console()


class TerminalRenderer:
    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose
        self._console = Console()

    def render(self, event: Event) -> None:
        match event:
            case TextEvent(text=text):
                self._render_text(text)
            case ToolEvent():
                self._render_tool(event)
            case TurnStartEvent():
                if self._verbose:
                    self._console.print(
                        f"[dim]--- turn {event.turn_id} ---[/dim]"
                    )
            case TurnEndEvent():
                self._render_turn_end(event)
            case CompactionEvent():
                if self._verbose:
                    self._console.print(
                        "[dim]context compacted[/dim]"
                    )
            case ErrorEvent(error=error):
                self._console.print(f"[bold red]Error:[/bold red] {error}")

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
        else:
            args_display = str(event.tool_args)[:80]

        if args_display:
            header.append(f" {args_display}", style="dim")

        if event.duration_ms > 0:
            header.append(f" ({event.duration_ms}ms)", style="dim")

        self._console.print(header)

        result_str = str(event.result) if event.result else ""
        if result_str and self._verbose:
            truncated = result_str[:500]
            if len(result_str) > 500:
                truncated += "..."
            self._console.print(
                Panel(truncated, border_style="dim", expand=False)
            )

    def _render_turn_end(self, event: TurnEndEvent) -> None:
        parts = []
        if event.tool_calls > 0:
            parts.append(f"{event.tool_calls} tool calls")
        if event.tokens_used > 0:
            parts.append(f"{event.tokens_used} tokens")
        if event.cost_usd > 0:
            parts.append(f"${event.cost_usd:.4f}")

        if parts:
            summary = " | ".join(parts)
            self._console.print(f"\n[dim]{summary}[/dim]")
