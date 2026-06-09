"""CLI entry point — interactive REPL and one-shot mode."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from calamar.config import Config
from calamar.events import ErrorEvent, TextEvent, ToolEvent
from calamar.loop import AgentLoop
from calamar.render import TerminalRenderer
from calamar.tools.defaults import create_default_registry

BANNER = """\
[bold]calamar[/bold] [dim]v0.2[/dim]
[dim]Type your message, or /help for commands. Ctrl+C to interrupt, Ctrl+D to exit.[/dim]
"""

HELP_TEXT = """
[bold]Commands:[/bold]
  /help          Show this help
  /clear         Clear conversation history
  /model         Show current model
  /model list    List available models
  /model <name>  Switch model
  /cost          Show session cost
  /verbose       Toggle verbose mode
  /undo          Undo last agent change
  /diff          Show uncommitted changes
  /branch        Show current branch
  /exit          Exit
"""


def _build_config(args: argparse.Namespace) -> Config:
    provider = args.provider or os.environ.get("CALAMAR_PROVIDER", "openai")

    if provider == "ducky":
        ducky_token = args.api_key or os.environ.get("AONE_TOKEN", "")
        base_url = args.base_url or os.environ.get(
            "AONE_BASE_URL", "https://ducky.code.alibaba-inc.com",
        )
        return Config(
            model=args.model,
            provider="ducky",
            base_url=base_url,
            ducky_token=ducky_token,
            project_root=os.getcwd(),
        )

    api_key = args.api_key or os.environ.get(
        "OPENAI_API_KEY",
        os.environ.get("ANTHROPIC_API_KEY", ""),
    )
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")

    return Config(
        model=args.model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        project_root=os.getcwd(),
    )


async def _run_once(prompt: str, config: Config, verbose: bool) -> None:
    renderer = TerminalRenderer(verbose=verbose)
    tools = create_default_registry(working_dir=config.project_root or ".")
    agent = AgentLoop(config, tools=tools)

    async for event in agent.run(prompt):
        renderer.render(event)


async def _run_repl(config: Config, verbose: bool) -> None:
    from pathlib import Path

    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from rich.console import Console

    con = Console()
    con.print(BANNER)

    history_dir = Path.home() / ".calamar"
    history_dir.mkdir(exist_ok=True)
    session = PromptSession(history=FileHistory(str(history_dir / "history")))

    renderer = TerminalRenderer(verbose=verbose)
    tools = create_default_registry(working_dir=config.project_root or ".")
    agent = AgentLoop(config, tools=tools)
    total_cost = 0.0

    while True:
        try:
            con.print()
            user_input = await session.prompt_async("> ")
        except (EOFError, KeyboardInterrupt):
            con.print("\n[dim]Goodbye.[/dim]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/exit", "/quit", "/q"):
                con.print("[dim]Goodbye.[/dim]")
                break
            elif cmd == "/help":
                con.print(HELP_TEXT)
                continue
            elif cmd == "/clear":
                agent.clear_history()
                con.print("[dim]History cleared.[/dim]")
                continue
            elif cmd == "/model" or cmd.startswith("/model "):
                parts = user_input.strip().split(maxsplit=1)
                if len(parts) == 1:
                    con.print(f"[dim]Current model: {config.model}[/dim]")
                elif parts[1].lower() == "list":
                    con.print("[dim]Fetching models...[/dim]")
                    models = await agent.provider.list_models()
                    if models:
                        for m in models:
                            marker = " *" if m == config.model else ""
                            con.print(f"[dim]  {m}{marker}[/dim]")
                    else:
                        con.print("[dim]Could not fetch model list.[/dim]")
                else:
                    config.model = parts[1]
                    con.print(f"[dim]Switched to: {config.model}[/dim]")
                continue
            elif cmd == "/cost":
                con.print(f"[dim]Session cost: ${total_cost:.4f}[/dim]")
                continue
            elif cmd == "/verbose":
                verbose = not verbose
                renderer = TerminalRenderer(verbose=verbose)
                state = "on" if verbose else "off"
                con.print(f"[dim]Verbose mode: {state}[/dim]")
                continue
            elif cmd in ("/undo", "/undo --all"):
                if cmd == "/undo --all":
                    reverted = await agent.git.undo_all()
                    if reverted:
                        names = ", ".join(r.sha for r in reverted)
                        con.print(f"[dim]Reverted {len(reverted)} commits: {names}[/dim]")
                    else:
                        con.print("[dim]No agent commits to undo.[/dim]")
                else:
                    reverted = await agent.git.undo_last()
                    if reverted:
                        con.print(f"[dim]Reverted {reverted.sha}: {reverted.message}[/dim]")
                    else:
                        con.print("[dim]No agent commits to undo.[/dim]")
                continue
            elif cmd == "/diff":
                diff_text = await agent.git.diff()
                if diff_text:
                    con.print(diff_text)
                else:
                    con.print("[dim]No uncommitted changes.[/dim]")
                continue
            elif cmd == "/branch":
                branch = await agent.git.current_branch()
                con.print(f"[dim]Branch: {branch or '(detached)'}[/dim]")
                continue

        try:
            async for event in agent.run(user_input):
                renderer.render(event)
                if hasattr(event, "cost_usd"):
                    total_cost += event.cost_usd
        except KeyboardInterrupt:
            con.print("\n[dim]Interrupted.[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="calamar",
        description="A general-purpose AI agent.",
    )
    parser.add_argument(
        "prompt", nargs="*", default=[],
        help="One-shot prompt (omit for interactive REPL)",
    )
    parser.add_argument(
        "--model", "-m", default="claude-sonnet-4-6-20250514",
        help="Model to use (default: claude-sonnet-4-6-20250514)",
    )
    parser.add_argument(
        "--provider", "-p", default="",
        help="LLM provider: openai (default) or ducky (or set CALAMAR_PROVIDER)",
    )
    parser.add_argument(
        "--api-key", "-k", default="",
        help="API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url", "-u", default="",
        help="API base URL (or set OPENAI_BASE_URL env var)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed tool output",
    )

    args = parser.parse_args()
    config = _build_config(args)

    if not config.api_key and config.provider != "ducky":
        print(
            "Error: No API key. Set OPENAI_API_KEY or use --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.prompt:
        prompt = " ".join(args.prompt)
        asyncio.run(_run_once(prompt, config, args.verbose))
    else:
        asyncio.run(_run_repl(config, args.verbose))


if __name__ == "__main__":
    main()
