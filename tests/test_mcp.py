"""Tests for MCP tool integration."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from calamar.tools.base import RawToolSpec
from calamar.tools.mcp import (
    McpConnection,
    McpManager,
    McpServerConfig,
    McpTool,
    _sanitize_name,
    load_mcp_config,
)
from calamar.tools.registry import ToolRegistry


class TestSanitizeName:
    def test_simple(self):
        assert _sanitize_name("hello") == "hello"

    def test_dashes_and_dots(self):
        assert _sanitize_name("my-tool.v2") == "my_tool_v2"

    def test_special_chars(self):
        assert _sanitize_name("foo@bar/baz") == "foo_bar_baz"

    def test_already_clean(self):
        assert _sanitize_name("read_file_123") == "read_file_123"


class TestLoadMcpConfig:
    def test_no_config_file(self):
        with tempfile.TemporaryDirectory() as d:
            assert load_mcp_config(d) == {}

    def test_valid_config(self):
        with tempfile.TemporaryDirectory() as d:
            config = {
                "mcpServers": {
                    "fs": {
                        "command": "npx",
                        "args": ["-y", "@mcp/server-filesystem", "/tmp"],
                        "env": {"DEBUG": "1"},
                    },
                    "git": {
                        "command": "mcp-git",
                    },
                },
            }
            with open(os.path.join(d, ".mcp.json"), "w") as f:
                json.dump(config, f)

            result = load_mcp_config(d)
            assert len(result) == 2
            assert result["fs"].command == "npx"
            assert result["fs"].args == ["-y", "@mcp/server-filesystem", "/tmp"]
            assert result["fs"].env == {"DEBUG": "1"}
            assert result["git"].command == "mcp-git"
            assert result["git"].args == []

    def test_invalid_json(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, ".mcp.json"), "w") as f:
                f.write("{invalid json")
            assert load_mcp_config(d) == {}

    def test_missing_command(self):
        with tempfile.TemporaryDirectory() as d:
            config = {
                "mcpServers": {
                    "bad": {"args": ["--flag"]},
                },
            }
            with open(os.path.join(d, ".mcp.json"), "w") as f:
                json.dump(config, f)
            assert load_mcp_config(d) == {}

    def test_non_dict_server_entry(self):
        with tempfile.TemporaryDirectory() as d:
            config = {"mcpServers": {"bad": "not-a-dict"}}
            with open(os.path.join(d, ".mcp.json"), "w") as f:
                json.dump(config, f)
            assert load_mcp_config(d) == {}


class TestMcpTool:
    def _make_tool(self, mcp_name: str = "read", server: str = "fs") -> McpTool:
        conn = MagicMock(spec=McpConnection)
        return McpTool(
            _mcp_name=mcp_name,
            _server_name=server,
            _description="Read a file",
            _input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            _connection=conn,
        )

    def test_spec_name_format(self):
        tool = self._make_tool("read-file", "my-server")
        assert tool.spec.name == "mcp_my_server_read_file"

    def test_spec_is_raw(self):
        tool = self._make_tool()
        assert isinstance(tool.spec, RawToolSpec)

    def test_openai_schema(self):
        tool = self._make_tool()
        schema = tool.spec.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "mcp_fs_read"
        assert "path" in schema["function"]["parameters"]["properties"]

    def test_execute_success(self):
        tool = self._make_tool()

        @dataclass
        class ContentBlock:
            text: str = "file contents here"

        @dataclass
        class CallResult:
            content: list[Any] = field(default_factory=lambda: [ContentBlock()])
            isError: bool = False

        tool._connection.call_tool = AsyncMock(return_value=CallResult())
        result = asyncio.run(tool.execute(path="/tmp/test.txt"))
        assert result.success
        assert "file contents here" in result.output
        assert result.metadata["mcp_server"] == "fs"

    def test_execute_error_result(self):
        tool = self._make_tool()

        @dataclass
        class ContentBlock:
            text: str = "not found"

        @dataclass
        class CallResult:
            content: list[Any] = field(default_factory=lambda: [ContentBlock()])
            isError: bool = True

        tool._connection.call_tool = AsyncMock(return_value=CallResult())
        result = asyncio.run(tool.execute(path="/nonexistent"))
        assert not result.success
        assert "not found" in result.error

    def test_execute_exception(self):
        tool = self._make_tool()
        tool._connection.call_tool = AsyncMock(
            side_effect=RuntimeError("connection lost"),
        )
        result = asyncio.run(tool.execute(path="/tmp/test.txt"))
        assert not result.success
        assert "connection lost" in result.error

    def test_execute_empty_content(self):
        tool = self._make_tool()

        @dataclass
        class CallResult:
            content: list[Any] = field(default_factory=list)
            isError: bool = False

        tool._connection.call_tool = AsyncMock(return_value=CallResult())
        result = asyncio.run(tool.execute(path="/tmp/test.txt"))
        assert result.success
        assert result.output == "(no output)"


class TestRawToolSpec:
    def test_passthrough_schema(self):
        spec = RawToolSpec(
            name="test",
            description="A test tool",
            raw_parameters={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        )
        schema = spec.to_openai_schema()
        assert schema["function"]["parameters"]["properties"]["items"]["type"] == "array"

    def test_empty_parameters(self):
        spec = RawToolSpec(name="test", description="No params")
        schema = spec.to_openai_schema()
        assert schema["function"]["parameters"] == {
            "type": "object",
            "properties": {},
        }


class TestMcpManager:
    def test_no_mcp_package(self):
        registry = ToolRegistry()
        with patch("calamar.tools.mcp._MCP_AVAILABLE", False):
            mgr = McpManager(registry, "/tmp")
            asyncio.run(self._enter_exit(mgr))
        assert len(registry) == 0

    def test_no_config_file(self):
        registry = ToolRegistry()
        with tempfile.TemporaryDirectory() as d:
            mgr = McpManager(registry, d)
            asyncio.run(self._enter_exit(mgr))
        assert len(registry) == 0
        assert mgr.server_count == 0
        assert mgr.tool_count == 0

    @staticmethod
    async def _enter_exit(mgr: McpManager) -> None:
        async with mgr:
            pass


class TestMcpServerConfig:
    def test_defaults(self):
        cfg = McpServerConfig(command="node")
        assert cfg.args == []
        assert cfg.env == {}

    def test_with_values(self):
        cfg = McpServerConfig(
            command="python",
            args=["-m", "server"],
            env={"PORT": "8080"},
        )
        assert cfg.command == "python"
        assert cfg.args == ["-m", "server"]
        assert cfg.env["PORT"] == "8080"
