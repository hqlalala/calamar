"""OpenAI-compatible provider — wraps AsyncOpenAI client."""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from calamar.config import Config
from calamar.events import TokenUsage
from calamar.providers import CompletionResponse, ToolCallData


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
