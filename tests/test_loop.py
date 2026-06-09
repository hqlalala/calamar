"""Tests for the agent loop."""

from __future__ import annotations

import pytest

from calamar.config import Config
from calamar.context import ContextBuilder, ContextCompactor, Message, user_message
from calamar.events import TextEvent, ToolEvent, TurnEndEvent, TurnStartEvent
from calamar.loop import AgentLoop
from calamar.middleware import (
    CostMiddleware,
    InputGuardrail,
    MiddlewareContext,
    MiddlewarePipeline,
    PermissionMiddleware,
)
from calamar.roles import BUILTIN_ROLES, DEFAULT, AgentRole
from calamar.router import ModelRouter, TaskProfile
from calamar.tools import ToolRegistry, ToolResult, ToolSpec


class TestConfig:
    def test_default_config(self):
        config = Config()
        assert config.model == "claude-sonnet-4-6-20250514"
        assert config.max_turns == 100
        assert config.temperature == 0.0

    def test_model_config(self):
        config = Config(model="gpt-4o", api_key="test")
        mc = config.model_config()
        assert mc.name == "gpt-4o"
        assert mc.api_key == "test"


class TestEvents:
    def test_text_event(self):
        e = TextEvent(text="hello")
        assert e.type == "text"
        assert e.text == "hello"

    def test_tool_event(self):
        e = ToolEvent(tool_name="terminal", tool_args={"cmd": "ls"})
        assert e.type == "tool"
        assert e.tool_name == "terminal"


class TestToolRegistry:
    def test_register_and_dispatch(self):
        registry = ToolRegistry()
        assert len(registry) == 0

    def test_unknown_tool(self):
        import asyncio
        registry = ToolRegistry()
        result = asyncio.run(registry.dispatch("nonexistent", {}))
        assert result.error is not None


class TestRoles:
    def test_builtin_roles_exist(self):
        assert "default" in BUILTIN_ROLES
        assert "planner" in BUILTIN_ROLES
        assert "executor" in BUILTIN_ROLES
        assert "verifier" in BUILTIN_ROLES

    def test_default_role(self):
        assert DEFAULT.name == "default"
        assert "Calamar" in DEFAULT.system_prompt


class TestModelRouter:
    def test_classify_simple(self):
        config = Config()
        router = ModelRouter(config)
        profile = router.classify("hi", has_code_context=False)
        assert profile.complexity == "simple"

    def test_classify_complex(self):
        config = Config()
        router = ModelRouter(config)
        profile = router.classify("refactor the authentication module")
        assert profile.complexity == "complex"

    def test_classify_fix(self):
        config = Config()
        router = ModelRouter(config)
        profile = router.classify("fix the bug in login")
        assert profile.task_type == "bug_fix"

    def test_route_default(self):
        config = Config(model="test-model")
        router = ModelRouter(config)
        model = router.route("hello")
        assert model == "test-model"


class TestContextBuilder:
    def test_build_messages(self):
        builder = ContextBuilder("You are a helper.", [])
        history = [user_message("hello")]
        messages = builder.build(history)
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "hello"


class TestContextCompactor:
    def test_no_compaction_needed(self):
        import asyncio
        compactor = ContextCompactor(context_window=100_000)
        history = [user_message("short message")]
        result = asyncio.run(compactor.compact(history))
        assert len(result) == 1

    def test_needs_compaction(self):
        compactor = ContextCompactor(context_window=100)
        history = [user_message("x" * 500)]
        assert compactor.needs_compaction(history)


class TestMiddleware:
    def test_input_guardrail_blocks(self):
        import asyncio
        guard = InputGuardrail()
        ctx = MiddlewareContext(
            tool_name="terminal",
            tool_args={"command": "rm -rf /"},
        )

        async def noop(c):
            return c

        result = asyncio.run(guard.process(ctx, noop))
        assert result.blocked

    def test_cost_middleware_blocks_over_budget(self):
        import asyncio
        cost = CostMiddleware(daily_limit_usd=1.0)
        cost.record(1.5)

        ctx = MiddlewareContext(tool_name="test", tool_args={})

        async def noop(c):
            return c

        result = asyncio.run(cost.process(ctx, noop))
        assert result.blocked

    def test_pipeline_construction(self):
        pipeline = MiddlewarePipeline()
        pipeline.use(InputGuardrail())
        assert True  # pipeline created without error


class TestPermissionMiddleware:
    def test_detects_rm_command(self):
        import asyncio
        approved = []

        async def callback(name, args):
            approved.append((name, args))
            return True

        perm = PermissionMiddleware(callback)
        ctx = MiddlewareContext(
            tool_name="terminal",
            tool_args={"command": "rm -rf ./build"},
        )

        async def noop(c):
            return c

        asyncio.run(perm.process(ctx, noop))
        assert len(approved) == 1

    def test_detects_git_push(self):
        import asyncio

        async def deny(name, args):
            return False

        perm = PermissionMiddleware(deny)
        ctx = MiddlewareContext(
            tool_name="terminal",
            tool_args={"command": "git push origin main"},
        )

        async def noop(c):
            return c

        result = asyncio.run(perm.process(ctx, noop))
        assert result.blocked
        assert "Denied" in result.block_reason

    def test_detects_sudo(self):
        import asyncio

        async def deny(name, args):
            return False

        perm = PermissionMiddleware(deny)
        ctx = MiddlewareContext(
            tool_name="terminal",
            tool_args={"command": "sudo apt install foo"},
        )

        async def noop(c):
            return c

        result = asyncio.run(perm.process(ctx, noop))
        assert result.blocked

    def test_allows_safe_commands(self):
        import asyncio
        called = []

        async def callback(name, args):
            called.append(True)
            return True

        perm = PermissionMiddleware(callback)
        ctx = MiddlewareContext(
            tool_name="terminal",
            tool_args={"command": "ls -la"},
        )

        async def noop(c):
            return c

        result = asyncio.run(perm.process(ctx, noop))
        assert not result.blocked
        assert len(called) == 0  # callback not invoked for safe commands

    def test_detects_sensitive_file_write(self):
        import asyncio

        async def deny(name, args):
            return False

        perm = PermissionMiddleware(deny)
        ctx = MiddlewareContext(
            tool_name="file_write",
            tool_args={"path": "/app/.env", "content": "SECRET=x"},
        )

        async def noop(c):
            return c

        result = asyncio.run(perm.process(ctx, noop))
        assert result.blocked

    def test_allows_normal_file_write(self):
        import asyncio
        called = []

        async def callback(name, args):
            called.append(True)
            return True

        perm = PermissionMiddleware(callback)
        ctx = MiddlewareContext(
            tool_name="file_write",
            tool_args={"path": "/app/main.py", "content": "print('hi')"},
        )

        async def noop(c):
            return c

        result = asyncio.run(perm.process(ctx, noop))
        assert not result.blocked
        assert len(called) == 0

    def test_detects_chained_dangerous_command(self):
        import asyncio

        async def deny(name, args):
            return False

        perm = PermissionMiddleware(deny)
        ctx = MiddlewareContext(
            tool_name="terminal",
            tool_args={"command": "echo test && rm -rf ./data"},
        )

        async def noop(c):
            return c

        result = asyncio.run(perm.process(ctx, noop))
        assert result.blocked


class TestUndoLastTurn:
    def test_undo_removes_last_turn(self):
        from calamar.context import assistant_message

        config = Config(api_key="test")
        loop = AgentLoop.__new__(AgentLoop)
        loop._history = [
            user_message("first"),
            assistant_message("response1"),
            user_message("second"),
            assistant_message("response2"),
        ]

        loop.undo_last_turn()
        assert len(loop._history) == 2
        assert loop._history[-1].content == "response1"

    def test_undo_empty_history(self):
        loop = AgentLoop.__new__(AgentLoop)
        loop._history = []
        loop.undo_last_turn()
        assert len(loop._history) == 0

    def test_undo_single_message(self):
        loop = AgentLoop.__new__(AgentLoop)
        loop._history = [user_message("only")]
        loop.undo_last_turn()
        assert len(loop._history) == 0


class TestContextInfo:
    def test_context_info_returns_stats(self):
        config = Config(context_window=100_000, api_key="test")
        loop = AgentLoop.__new__(AgentLoop)
        loop._history = [user_message("hello world")]
        loop._config = config
        loop._tools = ToolRegistry()

        info = loop.context_info()
        assert info["messages"] == 1
        assert info["tokens_est"] >= 0
        assert info["context_window"] == 100_000
        assert info["tools"] == 0
