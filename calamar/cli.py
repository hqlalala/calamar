"""Minimal CLI entry point."""

from __future__ import annotations

import asyncio
import sys

from calamar.config import Config
from calamar.events import ErrorEvent, TextEvent, ToolEvent
from calamar.loop import AgentLoop


async def _run(prompt: str) -> None:
    config = Config()
    agent = AgentLoop(config)

    async for event in agent.run(prompt):
        if isinstance(event, TextEvent):
            print(event.text)
        elif isinstance(event, ToolEvent):
            print(f"[tool:{event.tool_name}] {event.result}")
        elif isinstance(event, ErrorEvent):
            print(f"[error] {event.error}", file=sys.stderr)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: calamar <prompt>")
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])
    asyncio.run(_run(prompt))


if __name__ == "__main__":
    main()
