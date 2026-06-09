"""CLI entry point — interactive REPL and one-shot mode."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from calamar.config import Config
from calamar.events import ErrorEvent, TextEvent, ToolEvent
from calamar.loop import AgentLoop
from calamar.render import TerminalRenderer
from calamar.tools.defaults import create_default_registry

try:
    from calamar.tools.mcp import McpManager
except ImportError:
    from contextlib import asynccontextmanager as _acm

    @_acm
    async def McpManager(*_a, **_kw):  # type: ignore[misc]
        yield type("_Stub", (), {"server_count": 0, "tool_count": 0})()

BANNER = """\
[bold]calamar[/bold] [dim]v0.4[/dim]
[dim]Type your message, or /help for commands. Alt+Enter for newline, Ctrl+D to exit.[/dim]
"""

HELP_TEXT = """
[bold]Commands:[/bold]
  /help          Show this help
  /clear         Clear conversation history
  /retry         Retry last message
  /compact       Compact context to free tokens
  /context       Show context usage
  /model         Show current model
  /model list    List available models
  /model <name>  Switch model
  /cost          Show session cost
  /config        Show current configuration
  /verbose       Toggle verbose mode
  /undo          Undo last agent change
  /diff          Show uncommitted changes
  /branch        Show current branch
  /init          Initialize project config
  /exit          Exit
"""

_CONFIG_KEYS = ("model", "provider", "api_key", "base_url", "max_tokens", "temperature")


def _load_config_files() -> dict[str, str]:
    """Load config from ~/.calamar/config.json and .calamar/config.json."""
    merged: dict[str, str] = {}

    global_path = Path.home() / ".calamar" / "config.json"
    if global_path.is_file():
        try:
            with open(global_path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                merged.update({k: v for k, v in data.items() if k in _CONFIG_KEYS})
        except (json.JSONDecodeError, OSError):
            pass

    project_path = Path.cwd() / ".calamar" / "config.json"
    if project_path.is_file():
        try:
            with open(project_path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                merged.update({k: v for k, v in data.items() if k in _CONFIG_KEYS})
        except (json.JSONDecodeError, OSError):
            pass

    return merged


def _infer_provider(model: str) -> str:
    """Auto-detect provider from model name."""
    if model.startswith("claude-"):
        return "anthropic"
    return "openai"


def _build_config(args: argparse.Namespace) -> Config:
    file_cfg = _load_config_files()

    model = args.model
    if model == "claude-sonnet-4-6-20250514" and "model" in file_cfg:
        model = file_cfg["model"]

    provider = args.provider or os.environ.get(
        "CALAMAR_PROVIDER", file_cfg.get("provider", ""),
    )

    if provider == "ducky":
        ducky_token = args.api_key or os.environ.get("AONE_TOKEN", "")
        base_url = args.base_url or os.environ.get(
            "AONE_BASE_URL", "https://ducky.code.alibaba-inc.com",
        )
        return Config(
            model=model,
            provider="ducky",
            base_url=base_url,
            ducky_token=ducky_token,
            project_root=os.getcwd(),
        )

    if not provider:
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if anthropic_key or model.startswith("claude-"):
            provider = "anthropic"
        else:
            provider = _infer_provider(model)

    if provider == "anthropic":
        api_key = args.api_key or os.environ.get(
            "ANTHROPIC_API_KEY", file_cfg.get("api_key", ""),
        )
    else:
        api_key = args.api_key or os.environ.get(
            "OPENAI_API_KEY",
            os.environ.get("ANTHROPIC_API_KEY", file_cfg.get("api_key", "")),
        )

    base_url = args.base_url or os.environ.get(
        "OPENAI_BASE_URL", file_cfg.get("base_url"),
    )

    max_tokens = int(file_cfg.get("max_tokens", 8192))
    temperature = float(file_cfg.get("temperature", 0.0))

    return Config(
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        project_root=os.getcwd(),
    )


async def _run_once(prompt: str, config: Config, verbose: bool) -> None:
    renderer = TerminalRenderer(verbose=verbose)
    tools = create_default_registry(working_dir=config.project_root or ".")
    async with McpManager(tools, config.project_root or "."):
        agent = AgentLoop(config, tools=tools)
        async for event in agent.run(prompt):
            renderer.render(event)


async def _ask_permission(tool_name: str, tool_args: dict) -> bool:
    """Prompt the user for permission to run a dangerous operation."""
    if tool_name == "terminal":
        display = tool_args.get("command", str(tool_args)[:80])
    else:
        display = tool_args.get("path", str(tool_args)[:80])

    loop = asyncio.get_running_loop()
    try:
        answer = await loop.run_in_executor(
            None, lambda: input(f"\n  Allow {tool_name}: {display}? [y/N] "),
        )
    except (EOFError, KeyboardInterrupt):
        return False
    return answer.strip().lower() in ("y", "yes")


async def _run_repl(config: Config, verbose: bool) -> None:
    from pathlib import Path

    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from rich.console import Console

    class _CmdCompleter(Completer):
        _COMMANDS = (
            "/help", "/clear", "/retry", "/compact", "/context",
            "/model", "/model list", "/cost", "/config", "/verbose",
            "/undo", "/diff", "/branch", "/init", "/exit",
        )

        def get_completions(self, document, complete_event):
            text = document.text_before_cursor.lstrip()
            if not text.startswith("/"):
                return
            for cmd in self._COMMANDS:
                if cmd.startswith(text) and cmd != text:
                    yield Completion(cmd, start_position=-len(text))

    con = Console()
    con.print(BANNER)
    con.print(f"[dim]Model: {config.model} ({config.provider})[/dim]")

    history_dir = Path.home() / ".calamar"
    history_dir.mkdir(exist_ok=True)

    bindings = KeyBindings()

    @bindings.add("enter")
    def _submit(event):
        event.current_buffer.validate_and_handle()

    @bindings.add("escape", "enter")
    def _newline(event):
        event.current_buffer.insert_text("\n")

    session = PromptSession(
        history=FileHistory(str(history_dir / "history")),
        key_bindings=bindings,
        multiline=True,
        prompt_continuation="  ",
        completer=_CmdCompleter(),
    )

    renderer = TerminalRenderer(verbose=verbose)
    tools = create_default_registry(working_dir=config.project_root or ".")
    async with McpManager(tools, config.project_root or ".") as mcp:
        if mcp.server_count > 0:
            con.print(
                f"[dim]MCP: {mcp.server_count} server(s), "
                f"{mcp.tool_count} tool(s)[/dim]"
            )
        agent = AgentLoop(config, tools=tools, permission_callback=_ask_permission)
        total_cost = 0.0
        last_user_input = ""

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
                    last_user_input = ""
                    con.print("[dim]History cleared.[/dim]")
                    continue
                elif cmd == "/retry":
                    if not last_user_input:
                        con.print("[dim]Nothing to retry.[/dim]")
                        continue
                    agent.undo_last_turn()
                    user_input = last_user_input
                    con.print(f"[dim]Retrying: {user_input[:60]}...[/dim]"
                              if len(user_input) > 60
                              else f"[dim]Retrying: {user_input}[/dim]")
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
                elif cmd == "/config":
                    con.print(f"[dim]  Model:       {config.model}[/dim]")
                    con.print(f"[dim]  Provider:    {config.provider}[/dim]")
                    con.print(f"[dim]  Max tokens:  {config.max_tokens}[/dim]")
                    con.print(f"[dim]  Temperature: {config.temperature}[/dim]")
                    if config.base_url:
                        con.print(f"[dim]  Base URL:    {config.base_url}[/dim]")
                    api_display = config.api_key[:8] + "..." if len(config.api_key) > 8 else "(not set)"
                    con.print(f"[dim]  API key:     {api_display}[/dim]")
                    continue
                elif cmd == "/verbose":
                    verbose = not verbose
                    renderer = TerminalRenderer(verbose=verbose)
                    state = "on" if verbose else "off"
                    con.print(f"[dim]Verbose mode: {state}[/dim]")
                    continue
                elif cmd == "/compact":
                    removed = await agent.compact()
                    if removed > 0:
                        con.print(f"[dim]Compacted: removed {removed} messages.[/dim]")
                    else:
                        con.print("[dim]Nothing to compact.[/dim]")
                    continue
                elif cmd == "/context":
                    info = agent.context_info()
                    pct = info["tokens_est"] * 100 // max(info["context_window"], 1)
                    con.print(f"[dim]  Messages:  {info['messages']}[/dim]")
                    con.print(f"[dim]  Tokens:    ~{info['tokens_est']:,} / {info['context_window']:,} ({pct}%)[/dim]")
                    con.print(f"[dim]  Tools:     {info['tools']}[/dim]")
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
                elif cmd == "/init":
                    _init_project(config, con)
                    continue

            try:
                last_user_input = user_input
                async for event in agent.run(user_input):
                    renderer.render(event)
                    if hasattr(event, "cost_usd"):
                        total_cost += event.cost_usd
            except KeyboardInterrupt:
                renderer.flush()
                con.print("\n[dim]Interrupted.[/dim]")


_AGENT_MD_TEMPLATE = """\
# Project Instructions

<!-- Calamar reads this file to understand project-specific context. -->
<!-- Edit it to match your project's needs. -->

## Overview

Describe your project here.

## Code Style

- Language:
- Framework:
- Conventions:

## Important Notes

-
"""


def _init_project(config: Config, con) -> None:
    """Create .calamar/config.json and AGENT.md in the current directory."""
    project_dir = Path.cwd() / ".calamar"
    project_dir.mkdir(exist_ok=True)

    config_path = project_dir / "config.json"
    if config_path.exists():
        con.print("[dim].calamar/config.json already exists, skipping.[/dim]")
    else:
        cfg = {"model": config.model, "provider": config.provider}
        if config.base_url:
            cfg["base_url"] = config.base_url
        config_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        con.print(f"[dim]Created .calamar/config.json[/dim]")

    agent_md = Path.cwd() / "AGENT.md"
    if agent_md.exists():
        con.print("[dim]AGENT.md already exists, skipping.[/dim]")
    else:
        agent_md.write_text(_AGENT_MD_TEMPLATE, encoding="utf-8")
        con.print("[dim]Created AGENT.md — edit it with your project instructions.[/dim]")


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
