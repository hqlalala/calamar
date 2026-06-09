"""OpenAI-compatible provider — wraps AsyncOpenAI client."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from calamar.config import Config
from calamar.events import TokenUsage
from calamar.providers import CompletionResponse, StreamDelta, ToolCallData


class OpenAIProvider:
    """Provider for OpenAI and any OpenAI-compatible API (vLLM, Ollama, etc.)."""

    def __init__(self, config: Config) -> None:
        self._client = AsyncOpenAI(
            api_key=config.api_key or "not-set",
            base_url=config.base_url,
        )

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> CompletionResponse:
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        usage = self._extract_usage(response, model)

        if choice.finish_reason == "stop" or not choice.message.tool_calls:
            return CompletionResponse(
                text=choice.message.content or "",
                finish_reason="stop",
                usage=usage,
            )

        tool_calls = []
        for tc in choice.message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCallData(
                id=tc.id,
                name=tc.function.name,
                arguments=args,
            ))

        return CompletionResponse(
            text=choice.message.content or "",
            tool_calls=tool_calls,
            finish_reason="tool_calls",
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
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        tc_id_map: dict[int, str] = {}
        tc_name_map: dict[int, str] = {}
        tc_args_map: dict[int, str] = {}
        finish_reason: str | None = None

        async for chunk in response:
            if not chunk.choices:
                usage = self._extract_usage(chunk, model)
                if usage.total_tokens > 0:
                    yield StreamDelta(usage=usage)
                continue

            delta = chunk.choices[0].delta
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

            if delta.content:
                yield StreamDelta(text=delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if tc_delta.id:
                        tc_id_map[idx] = tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        tc_name_map[idx] = tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        tc_args_map[idx] = (
                            tc_args_map.get(idx, "") + tc_delta.function.arguments
                        )

        tool_calls = []
        for idx in sorted(tc_id_map):
            raw_args = tc_args_map.get(idx, "{}")
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCallData(
                id=tc_id_map[idx],
                name=tc_name_map.get(idx, ""),
                arguments=args,
            ))

        fr = "tool_calls" if tool_calls else (finish_reason or "stop")
        yield StreamDelta(tool_calls=tool_calls or None, finish_reason=fr)

    def _extract_usage(self, response: Any, model: str) -> TokenUsage:
        usage = response.usage
        if usage is None:
            return TokenUsage(model=model)

        token_usage = TokenUsage(
            prompt_tokens=usage.prompt_tokens or 0,
            completion_tokens=usage.completion_tokens or 0,
            total_tokens=usage.total_tokens or 0,
            model=model,
        )
        cache = getattr(usage, "prompt_tokens_details", None)
        if cache:
            token_usage.cache_read_tokens = getattr(cache, "cached_tokens", 0)
        return token_usage

    async def list_models(self) -> list[str]:
        try:
            models = await self._client.models.list()
            return sorted(m.id for m in models.data)
        except Exception:
            return []
