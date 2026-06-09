"""Tests for context building and compaction."""

from __future__ import annotations

import pytest

from calamar.context import (
    ContextBuilder,
    ContextCompactor,
    Message,
    assistant_message,
    system_message,
    tool_result_message,
    user_message,
)


class TestMessageHelpers:
    def test_user_message(self):
        msg = user_message("hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_assistant_message(self):
        msg = assistant_message("hi")
        assert msg.role == "assistant"
        assert msg.content == "hi"

    def test_system_message(self):
        msg = system_message("you are helpful")
        assert msg.role == "system"

    def test_tool_result_message(self):
        msg = tool_result_message("tc_1", "output", name="file_read")
        assert msg.role == "tool"
        assert msg.tool_call_id == "tc_1"
        assert msg.content == "output"
        assert msg.name == "file_read"

    def test_tool_result_no_name(self):
        msg = tool_result_message("tc_1", "output")
        assert msg.name is None

    def test_tool_result_empty_name(self):
        msg = tool_result_message("tc_1", "output", name="")
        assert msg.name is None


class TestContextBuilder:
    def test_builds_system_message(self):
        builder = ContextBuilder(
            system_prompt="You are helpful.",
            tool_schemas=[],
        )
        msgs = builder.build([])
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"
        assert "You are helpful." in msgs[0]["content"]

    def test_includes_history(self):
        builder = ContextBuilder(system_prompt="sys", tool_schemas=[])
        history = [user_message("hi"), assistant_message("hello")]
        msgs = builder.build(history)
        assert len(msgs) == 3
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_includes_context_files(self):
        builder = ContextBuilder(system_prompt="base", tool_schemas=[])
        builder.add_context_file("# Repo Map\nfile1.py")
        builder.add_context_file("# Agent Config\nrule1")
        msgs = builder.build([])
        system_content = msgs[0]["content"]
        assert "Repo Map" in system_content
        assert "Agent Config" in system_content

    def test_preserves_tool_call_id(self):
        builder = ContextBuilder(system_prompt="sys", tool_schemas=[])
        msg = tool_result_message("tc_123", "result")
        msgs = builder.build([msg])
        assert msgs[1]["tool_call_id"] == "tc_123"

    def test_preserves_tool_calls(self):
        builder = ContextBuilder(system_prompt="sys", tool_schemas=[])
        tc = [{"id": "tc_1", "type": "function", "function": {"name": "ls"}}]
        msg = Message(role="assistant", content="let me check", tool_calls=tc)
        msgs = builder.build([msg])
        assert msgs[1]["tool_calls"] == tc

    def test_preserves_name(self):
        builder = ContextBuilder(system_prompt="sys", tool_schemas=[])
        msg = tool_result_message("tc_1", "output", name="terminal")
        msgs = builder.build([msg])
        assert msgs[1]["name"] == "terminal"

    def test_tool_schemas_property(self):
        schemas = [{"type": "function", "function": {"name": "test"}}]
        builder = ContextBuilder(system_prompt="sys", tool_schemas=schemas)
        assert builder.tool_schemas == schemas


class TestContextCompactor:
    def test_no_compaction_needed(self):
        compactor = ContextCompactor(context_window=100_000)
        history = [user_message("hi"), assistant_message("hello")]
        assert not compactor.needs_compaction(history)

    async def test_compact_noop_when_under_limit(self):
        compactor = ContextCompactor(context_window=100_000)
        history = [user_message("hi"), assistant_message("hello")]
        result = await compactor.compact(history)
        assert len(result) == 2

    async def test_truncate_tool_outputs(self):
        compactor = ContextCompactor(context_window=100, threshold=0.01)
        long_output = "x" * 3000
        history = [
            user_message("do something"),
            Message(role="tool", content=long_output, tool_call_id="tc_1"),
        ]
        result = await compactor.compact(history)
        tool_msg = [m for m in result if m.role == "tool"][0]
        assert len(tool_msg.content) < 3000
        assert "truncated" in tool_msg.content

    async def test_summarize_early_turns(self):
        # 10 messages × ~42 chars ÷ 3 ≈ 140 tokens.
        # context_window=200, threshold=0.5 → limit=100 → triggers compaction.
        # After summarization: summary + 4 recent ≈ 70 tokens → fits.
        compactor = ContextCompactor(context_window=200, threshold=0.5)
        history = [
            user_message(f"message number {i} with some padding text") for i in range(10)
        ]
        result = await compactor.compact(history)
        assert len(result) < len(history)
        assert result[0].role == "system"
        assert "Summary" in result[0].content

    async def test_summarize_with_custom_summarizer(self):
        compactor = ContextCompactor(context_window=200, threshold=0.5)
        history = [
            user_message(f"msg {i} with extra text to push over limit") for i in range(10)
        ]

        async def summarizer(msgs):
            return f"Custom summary of {len(msgs)} messages"

        result = await compactor.compact(history, summarizer=summarizer)
        assert "Custom summary" in result[0].content

    async def test_emergency_compact(self):
        compactor = ContextCompactor(context_window=10, threshold=0.01)
        history = [
            user_message("x" * 100) for _ in range(20)
        ]
        result = await compactor.compact(history)
        assert len(result) <= 3

    def test_needs_compaction_true(self):
        compactor = ContextCompactor(context_window=100, threshold=0.5)
        history = [user_message("x" * 300)]
        assert compactor.needs_compaction(history)

    async def test_preserves_recent_messages(self):
        compactor = ContextCompactor(context_window=200, threshold=0.01)
        history = [user_message(f"msg {i}") for i in range(8)]
        result = await compactor.compact(history)
        last_content = result[-1].content
        assert last_content == "msg 7"
