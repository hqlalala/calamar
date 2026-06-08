"""Context building and progressive compaction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str  # system / user / assistant / tool
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def system_message(content: str) -> Message:
    return Message(role="system", content=content)


def user_message(content: str) -> Message:
    return Message(role="user", content=content)


def assistant_message(content: str) -> Message:
    return Message(role="assistant", content=content)


def tool_result_message(tool_call_id: str, content: str) -> Message:
    return Message(role="tool", content=content, tool_call_id=tool_call_id)


class ContextBuilder:
    """Assemble context with static prefix + dynamic suffix for cache hits."""

    def __init__(self, system_prompt: str, tool_schemas: list[dict[str, Any]]) -> None:
        self._system_prompt = system_prompt
        self._tool_schemas = tool_schemas
        self._context_files: list[str] = []

    def add_context_file(self, content: str) -> None:
        self._context_files.append(content)

    def build(
        self,
        history: list[Message],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []

        system_parts = [self._system_prompt]
        if self._context_files:
            system_parts.append("\n---\n".join(self._context_files))
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        for msg in history:
            entry: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                entry["tool_calls"] = msg.tool_calls
            if msg.name:
                entry["name"] = msg.name
            messages.append(entry)

        return messages

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        return self._tool_schemas


class ContextCompactor:
    """Progressive 4-level compaction to stay within context window."""

    def __init__(self, context_window: int, threshold: float = 0.85) -> None:
        self._limit = int(context_window * threshold)

    async def compact(
        self, history: list[Message], summarizer: Any = None
    ) -> list[Message]:
        token_est = self._estimate_tokens(history)
        if token_est < self._limit:
            return history

        history = self._truncate_tool_outputs(history)
        if self._estimate_tokens(history) < self._limit:
            return history

        history = await self._summarize_early_turns(history, summarizer)
        if self._estimate_tokens(history) < self._limit:
            return history

        history = self._emergency_compact(history)
        return history

    def needs_compaction(self, history: list[Message]) -> bool:
        return self._estimate_tokens(history) >= self._limit

    @staticmethod
    def _estimate_tokens(messages: list[Message]) -> int:
        return sum(len(m.content) // 3 for m in messages)

    @staticmethod
    def _truncate_tool_outputs(history: list[Message]) -> list[Message]:
        result = []
        for msg in history:
            if msg.role == "tool" and len(msg.content) > 2000:
                truncated = msg.content[:1500] + "\n...[truncated]..."
                result.append(Message(
                    role=msg.role,
                    content=truncated,
                    tool_call_id=msg.tool_call_id,
                ))
            else:
                result.append(msg)
        return result

    @staticmethod
    async def _summarize_early_turns(
        history: list[Message], summarizer: Any = None,
    ) -> list[Message]:
        if len(history) <= 6:
            return history

        keep_recent = 4
        early = history[:-keep_recent]
        recent = history[-keep_recent:]

        if summarizer is not None:
            summary_text = await summarizer(early)
        else:
            turns = len(early)
            summary_text = f"[Summary of {turns} earlier messages omitted for context]"

        summary = Message(role="system", content=summary_text)
        return [summary] + recent

    @staticmethod
    def _emergency_compact(history: list[Message]) -> list[Message]:
        if len(history) <= 3:
            return history
        return history[-3:]
