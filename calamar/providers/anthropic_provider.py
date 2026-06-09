"""Native Anthropic provider — prompt caching, streaming, correct tool format."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic

from calamar.config import Config
from calamar.events import TokenUsage
from calamar.providers import CompletionResponse, StreamDelta, ToolCallData

_PRICING: dict[str, tuple[float, float, float, float]] = {
    # (input/M, output/M, cache_write/M, cache_read/M)
    "claude-sonnet-4-6-20250514": (3.0, 15.0, 3.75, 0.30),
    "claude-opus-4-6-20250205": (15.0, 75.0, 18.75, 1.50),
    "claude-haiku-4-5-20251001": (0.80, 4.0, 1.0, 0.08),
}
_DEFAULT_PRICING = (3.0, 15.0, 3.75, 0.30)


class AnthropicProvider:
    """Provider using the native Anthropic Messages API with prompt caching."""

    def __init__(self, config: Config) -> None:
        self._client = AsyncAnthropic(api_key=config.api_key or "not-set")

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> CompletionResponse:
        system, converted = self._convert_messages(messages)
        tool_defs = self._convert_tools(tools) if tools else []

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": converted,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        if tool_defs:
            kwargs["tools"] = tool_defs

        response = await self._client.messages.create(**kwargs)
        usage = self._extract_usage(response.usage, model)

        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCallData(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        finish = "tool_calls" if tool_calls else "stop"
        return CompletionResponse(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            finish_reason=finish,
            usage=usage,
        )

    async def stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> AsyncIterator[StreamDelta]:
        system, converted = self._convert_messages(messages)
        tool_defs = self._convert_tools(tools) if tools else []

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": converted,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        if tool_defs:
            kwargs["tools"] = tool_defs

        tool_inputs: dict[int, str] = {}
        tool_ids: dict[int, str] = {}
        tool_names: dict[int, str] = {}
        block_idx = 0

        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    block_idx = event.index
                    if block.type == "tool_use":
                        tool_ids[block_idx] = block.id
                        tool_names[block_idx] = block.name
                        tool_inputs[block_idx] = ""

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield StreamDelta(text=delta.text)
                    elif delta.type == "input_json_delta":
                        tool_inputs[block_idx] = (
                            tool_inputs.get(block_idx, "") + delta.partial_json
                        )

            final_message = stream.get_final_message()

        tool_calls = []
        for idx in sorted(tool_ids):
            raw = tool_inputs.get(idx, "{}")
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCallData(
                id=tool_ids[idx],
                name=tool_names.get(idx, ""),
                arguments=args,
            ))

        usage = self._extract_usage(final_message.usage, model)
        finish = "tool_calls" if tool_calls else "stop"
        yield StreamDelta(
            tool_calls=tool_calls or None,
            finish_reason=finish,
            usage=usage,
        )

    async def list_models(self) -> list[str]:
        try:
            result = await self._client.models.list()
            return sorted(m.id for m in result.data)
        except Exception:
            return sorted(_PRICING.keys())

    @staticmethod
    def _convert_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split system message and convert OpenAI format to Anthropic format."""
        system_parts: list[dict[str, Any]] = []
        converted: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")

            if role == "system":
                system_parts.append({
                    "type": "text",
                    "text": msg["content"],
                    "cache_control": {"type": "ephemeral"},
                })

            elif role == "assistant":
                content: list[dict[str, Any]] = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                for tc in msg.get("tool_calls", []) or []:
                    fn = tc.get("function", {})
                    raw_args = fn.get("arguments", "{}")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        args = {}
                    content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": fn["name"],
                        "input": args,
                    })
                converted.append({"role": "assistant", "content": content or msg.get("content", "")})

            elif role == "tool":
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg["tool_call_id"],
                        "content": msg["content"],
                    }],
                })

            elif role == "user":
                converted.append({"role": "user", "content": msg["content"]})

        converted = _merge_consecutive_user(converted)
        return system_parts, converted

    @staticmethod
    def _convert_tools(
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic tool format with cache control."""
        result = []
        for i, tool in enumerate(tools):
            fn = tool.get("function", tool)
            entry: dict[str, Any] = {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
            if i == len(tools) - 1:
                entry["cache_control"] = {"type": "ephemeral"}
            result.append(entry)
        return result

    @staticmethod
    def _extract_usage(usage: Any, model: str) -> TokenUsage:
        if usage is None:
            return TokenUsage(model=model)

        input_tok = getattr(usage, "input_tokens", 0) or 0
        output_tok = getattr(usage, "output_tokens", 0) or 0
        cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

        pricing = _PRICING.get(model, _DEFAULT_PRICING)
        cost = (
            (input_tok - cache_create - cache_read) * pricing[0]
            + output_tok * pricing[1]
            + cache_create * pricing[2]
            + cache_read * pricing[3]
        ) / 1_000_000

        return TokenUsage(
            prompt_tokens=input_tok,
            completion_tokens=output_tok,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_create,
            total_tokens=input_tok + output_tok,
            model=model,
            cost_usd=cost,
        )


def _merge_consecutive_user(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic forbids consecutive user messages; merge them."""
    if not messages:
        return messages
    merged: list[dict[str, Any]] = [messages[0]]
    for msg in messages[1:]:
        if msg["role"] == "user" and merged[-1]["role"] == "user":
            prev_content = merged[-1]["content"]
            curr_content = msg["content"]
            if isinstance(prev_content, list) and isinstance(curr_content, list):
                merged[-1]["content"] = prev_content + curr_content
            elif isinstance(prev_content, list):
                merged[-1]["content"] = prev_content + [{"type": "text", "text": curr_content}]
            elif isinstance(curr_content, list):
                merged[-1]["content"] = [{"type": "text", "text": prev_content}] + curr_content
            else:
                merged[-1]["content"] = f"{prev_content}\n\n{curr_content}"
        else:
            merged.append(msg)
    return merged
