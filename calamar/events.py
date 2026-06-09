"""Event types emitted by the agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(kw_only=True)
class Event:
    type: str


@dataclass(kw_only=True)
class TextEvent(Event):
    text: str
    type: str = "text"


@dataclass(kw_only=True)
class ToolEvent(Event):
    tool_name: str
    tool_args: dict[str, Any]
    result: Any = None
    duration_ms: int = 0
    type: str = "tool"


@dataclass
class CompactionEvent(Event):
    from_messages: int = 0
    to_messages: int = 0
    type: str = "compaction"


@dataclass
class ErrorEvent(Event):
    error: str = ""
    recoverable: bool = True
    type: str = "error"


@dataclass
class TurnStartEvent(Event):
    turn_id: str = ""
    type: str = "turn_start"


@dataclass
class TurnEndEvent(Event):
    turn_id: str = ""
    tool_calls: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    type: str = "turn_end"


@dataclass
class SteeringEvent(Event):
    instruction: str = ""
    type: str = "steering"


@dataclass
class RoleChangeEvent(Event):
    from_role: str = ""
    to_role: str = ""
    type: str = "role_change"


@dataclass
class CheckpointEvent(Event):
    checkpoint_id: str = ""
    type: str = "checkpoint"


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
