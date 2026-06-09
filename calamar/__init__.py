"""Calamar — a modular AI agent engine."""

from __future__ import annotations

__version__ = "0.5.0"

from calamar.config import Config
from calamar.events import Event, TextEvent, ToolEvent
from calamar.loop import AgentLoop
from calamar.roles import AgentRole

__all__ = [
    "AgentLoop",
    "AgentRole",
    "Config",
    "Event",
    "TextEvent",
    "ToolEvent",
]
