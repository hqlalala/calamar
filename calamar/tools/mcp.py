"""MCP (Model Context Protocol) tool integration."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from calamar.tools.base import RawToolSpec, ToolResult
from calamar.tools.registry import ToolRegistry

try:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False

logger = logging.getLogger(__name__)

_MCP_CONFIG_FILE = ".mcp.json"


def _sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", text)


@dataclass
class McpServerConfig:
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


def load_mcp_config(project_root: str) -> dict[str, McpServerConfig]:
    """Load .mcp.json from project root."""
    path = os.path.join(project_root, _MCP_CONFIG_FILE)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    servers = raw.get("mcpServers", {})
    result = {}
    for name, cfg in servers.items():
        if not isinstance(cfg, dict) or not cfg.get("command"):
            continue
        result[name] = McpServerConfig(
            command=cfg["command"],
            args=cfg.get("args", []),
            env=cfg.get("env", {}),
        )
    return result


class McpConnection:
    """Lifecycle manager for a single MCP stdio server connection."""

    def __init__(self, name: str, config: McpServerConfig) -> None:
        self.name = name
        self._config = config
        self._session: Any = None
        self._tools: list[Any] = []
        self._task: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._shutdown = asyncio.Event()
        self._error: Exception | None = None

    @property
    def tools(self) -> list[Any]:
        return self._tools

    @property
    def session(self) -> Any:
        return self._session

    async def connect(self, timeout: float = 30.0) -> None:
        self._task = asyncio.create_task(self._run())
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self._shutdown.set()
            raise TimeoutError(
                f"MCP server '{self.name}' did not start within {timeout}s"
            ) from None
        if self._error:
            raise self._error

    async def _run(self) -> None:
        env = {**os.environ, **self._config.env} if self._config.env else None
        server_params = StdioServerParameters(
            command=self._config.command,
            args=self._config.args,
            env=env,
        )
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    self._session = session
                    self._tools = result.tools if hasattr(result, "tools") else []
                    self._ready.set()
                    await self._shutdown.wait()
        except Exception as exc:
            self._error = exc
            self._ready.set()
        finally:
            self._session = None

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if not self._session:
            raise RuntimeError(f"MCP server '{self.name}' is not connected")
        return await self._session.call_tool(tool_name, arguments=arguments)

    async def close(self) -> None:
        self._shutdown.set()
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()


@dataclass
class McpTool:
    """Adapts an MCP tool to calamar's Tool protocol."""

    _mcp_name: str
    _server_name: str
    _description: str
    _input_schema: dict[str, Any]
    _connection: McpConnection

    @property
    def spec(self) -> RawToolSpec:
        return RawToolSpec(
            name=f"mcp_{_sanitize_name(self._server_name)}_{_sanitize_name(self._mcp_name)}",
            description=self._description or f"MCP tool '{self._mcp_name}'",
            raw_parameters=self._input_schema or {"type": "object", "properties": {}},
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            result = await self._connection.call_tool(self._mcp_name, kwargs)
        except Exception as exc:
            return ToolResult(error=f"MCP call failed: {exc}")

        if getattr(result, "isError", False):
            error_text = ""
            for block in getattr(result, "content", []) or []:
                if hasattr(block, "text"):
                    error_text += block.text
            return ToolResult(error=error_text or "MCP tool returned an error")

        parts = []
        for block in getattr(result, "content", []) or []:
            if hasattr(block, "text") and block.text:
                parts.append(block.text)
        return ToolResult(
            output="\n".join(parts) if parts else "(no output)",
            metadata={"mcp_server": self._server_name},
        )


class McpManager:
    """Connect to MCP servers and register their tools.

    Usage:
        async with McpManager(registry, project_root) as mcp:
            # registry now contains MCP tools
            ...
        # connections are closed
    """

    def __init__(self, registry: ToolRegistry, project_root: str = ".") -> None:
        self._registry = registry
        self._project_root = project_root
        self._connections: list[McpConnection] = []

    async def __aenter__(self) -> McpManager:
        if not _MCP_AVAILABLE:
            return self

        configs = load_mcp_config(self._project_root)
        if not configs:
            return self

        for name, config in configs.items():
            conn = McpConnection(name, config)
            try:
                await conn.connect()
            except Exception as exc:
                logger.warning("MCP server '%s' failed: %s", name, exc)
                continue

            self._connections.append(conn)
            for mcp_tool in conn.tools:
                tool = McpTool(
                    _mcp_name=mcp_tool.name,
                    _server_name=name,
                    _description=getattr(mcp_tool, "description", "") or "",
                    _input_schema=getattr(mcp_tool, "inputSchema", None) or {},
                    _connection=conn,
                )
                self._registry.register(tool)

        return self

    async def __aexit__(self, *exc: Any) -> None:
        for conn in self._connections:
            try:
                await conn.close()
            except Exception:
                pass

    @property
    def server_count(self) -> int:
        return len(self._connections)

    @property
    def tool_count(self) -> int:
        return sum(len(c.tools) for c in self._connections)
