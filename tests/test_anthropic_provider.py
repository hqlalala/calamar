"""Tests for the Anthropic provider."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from calamar.providers import create_provider
from calamar.providers.anthropic_provider import (
    AnthropicProvider,
    _merge_consecutive_user,
)


class TestMessageConversion:
    def test_system_extracted(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        system, converted = AnthropicProvider._convert_messages(messages)
        assert len(system) == 1
        assert system[0]["text"] == "You are helpful."
        assert system[0]["cache_control"] == {"type": "ephemeral"}
        assert len(converted) == 1
        assert converted[0]["role"] == "user"

    def test_tool_calls_converted(self):
        messages = [
            {"role": "user", "content": "Read file"},
            {
                "role": "assistant",
                "content": "I'll read it.",
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "file_read",
                        "arguments": '{"path": "/tmp/test.py"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "file contents here",
            },
        ]
        system, converted = AnthropicProvider._convert_messages(messages)
        assert len(system) == 0

        assistant_msg = converted[1]
        assert assistant_msg["role"] == "assistant"
        assert len(assistant_msg["content"]) == 2
        assert assistant_msg["content"][0]["type"] == "text"
        assert assistant_msg["content"][1]["type"] == "tool_use"
        assert assistant_msg["content"][1]["id"] == "call_123"
        assert assistant_msg["content"][1]["input"] == {"path": "/tmp/test.py"}

        tool_msg = converted[2]
        assert tool_msg["role"] == "user"
        assert tool_msg["content"][0]["type"] == "tool_result"
        assert tool_msg["content"][0]["tool_use_id"] == "call_123"

    def test_consecutive_user_messages_merged(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        _, converted = AnthropicProvider._convert_messages(messages)
        assert len(converted) == 1
        assert "first" in converted[0]["content"]
        assert "second" in converted[0]["content"]


class TestToolConversion:
    def test_openai_to_anthropic_format(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Run a command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                    },
                    "required": ["command"],
                },
            },
        }]
        result = AnthropicProvider._convert_tools(tools)
        assert len(result) == 1
        assert result[0]["name"] == "terminal"
        assert result[0]["input_schema"]["properties"]["command"]["type"] == "string"
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_cache_control_on_last_only(self):
        tools = [
            {"type": "function", "function": {"name": "a", "description": "A", "parameters": {}}},
            {"type": "function", "function": {"name": "b", "description": "B", "parameters": {}}},
        ]
        result = AnthropicProvider._convert_tools(tools)
        assert "cache_control" not in result[0]
        assert result[1]["cache_control"] == {"type": "ephemeral"}


class TestUsageExtraction:
    def test_with_cache(self):
        usage = MagicMock()
        usage.input_tokens = 1000
        usage.output_tokens = 500
        usage.cache_creation_input_tokens = 200
        usage.cache_read_input_tokens = 300

        result = AnthropicProvider._extract_usage(usage, "claude-sonnet-4-6-20250514")
        assert result.prompt_tokens == 1000
        assert result.completion_tokens == 500
        assert result.cache_read_tokens == 300
        assert result.cache_write_tokens == 200
        assert result.cost_usd > 0

    def test_none_usage(self):
        result = AnthropicProvider._extract_usage(None, "claude-sonnet-4-6-20250514")
        assert result.total_tokens == 0
        assert result.cost_usd == 0.0


class TestMergeConsecutiveUser:
    def test_no_merge_needed(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = _merge_consecutive_user(messages)
        assert len(result) == 2

    def test_string_merge(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        result = _merge_consecutive_user(messages)
        assert len(result) == 1
        assert "first" in result[0]["content"]
        assert "second" in result[0]["content"]

    def test_list_merge(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "a"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "b"}]},
        ]
        result = _merge_consecutive_user(messages)
        assert len(result) == 1
        assert len(result[0]["content"]) == 2

    def test_mixed_merge(self):
        messages = [
            {"role": "user", "content": "plain text"},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "r"}]},
        ]
        result = _merge_consecutive_user(messages)
        assert len(result) == 1
        assert isinstance(result[0]["content"], list)

    def test_empty(self):
        assert _merge_consecutive_user([]) == []


class TestCreateProvider:
    def test_anthropic_provider_created(self):
        config = MagicMock()
        config.provider = "anthropic"
        config.api_key = "test-key"
        provider = create_provider(config)
        assert isinstance(provider, AnthropicProvider)

    def test_openai_still_default(self):
        from calamar.providers.openai_provider import OpenAIProvider

        config = MagicMock()
        config.provider = "openai"
        config.api_key = "test-key"
        config.base_url = None
        provider = create_provider(config)
        assert isinstance(provider, OpenAIProvider)
