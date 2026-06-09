"""Provider abstraction for LLM backends."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from calamar.events import TokenUsage


@dataclass
class ToolCallData:
    """A parsed tool call from the model response."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class CompletionResponse:
    """Standardized response from any LLM provider."""

    text: str
    tool_calls: list[ToolCallData] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class StreamDelta:
    """A single chunk from a streaming LLM response."""

    text: str = ""
    tool_calls: list[ToolCallData] | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None


class Provider(Protocol):
    """Interface that all LLM providers must satisfy."""

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> CompletionResponse: ...

    def stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> AsyncIterator[StreamDelta]: ...


def create_provider(config: Any) -> Provider:
    """Instantiate a provider based on config.provider field."""
    provider = getattr(config, "provider", "openai")
    if provider == "ducky":
        from calamar.providers.ducky_provider import DuckyProvider

        return DuckyProvider(config)

    from calamar.providers.openai_provider import OpenAIProvider

    return OpenAIProvider(config)


def make_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


__all__ = [
    "CompletionResponse",
    "Provider",
    "StreamDelta",
    "ToolCallData",
    "create_provider",
    "make_tool_call_id",
]
