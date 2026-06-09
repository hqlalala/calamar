"""Middleware pipeline — layered processing for every tool call."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, Protocol

from calamar.tools.base import ToolResult


@dataclass
class MiddlewareContext:
    tool_name: str
    tool_args: dict[str, Any]
    tool_call_id: str = ""
    result: ToolResult | None = None
    blocked: bool = False
    block_reason: str = ""
    start_time_ns: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def block(self, reason: str) -> None:
        self.blocked = True
        self.block_reason = reason
        self.result = ToolResult(error=f"Blocked: {reason}")


class Middleware(Protocol):
    async def process(
        self, ctx: MiddlewareContext, call_next: CallNext,
    ) -> MiddlewareContext: ...


type CallNext = Any  # Callable[[MiddlewareContext], Awaitable[MiddlewareContext]]

PermissionCallback = Callable[[str, dict[str, Any]], Coroutine[Any, Any, bool]]


class MiddlewarePipeline:
    def __init__(self) -> None:
        self._layers: list[Middleware] = []

    def use(self, layer: Middleware) -> MiddlewarePipeline:
        self._layers.append(layer)
        return self

    async def execute(
        self, tool_name: str, tool_args: dict[str, Any], tool_call_id: str,
        executor: Any,
    ) -> MiddlewareContext:
        ctx = MiddlewareContext(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=tool_call_id,
            start_time_ns=time.monotonic_ns(),
        )

        async def run_executor(c: MiddlewareContext) -> MiddlewareContext:
            if c.blocked:
                return c
            c.result = await executor(c.tool_name, c.tool_args)
            return c

        chain = run_executor
        for layer in reversed(self._layers):
            chain = self._wrap(layer, chain)

        return await chain(ctx)

    @staticmethod
    def _wrap(layer: Middleware, next_fn: Any) -> Any:
        async def wrapper(ctx: MiddlewareContext) -> MiddlewareContext:
            return await layer.process(ctx, next_fn)
        return wrapper


class InputGuardrail:
    """Block dangerous inputs before they reach tools."""

    BLOCKED_PATTERNS = [
        "rm -rf /",
        "DROP TABLE",
        "format c:",
        "; curl ",
    ]

    async def process(
        self, ctx: MiddlewareContext, call_next: Any,
    ) -> MiddlewareContext:
        args_str = str(ctx.tool_args)
        for pattern in self.BLOCKED_PATTERNS:
            if pattern.lower() in args_str.lower():
                ctx.block(f"Dangerous pattern detected: {pattern}")
                return ctx
        return await call_next(ctx)


class PermissionMiddleware:
    """Ask user confirmation before executing dangerous operations.

    Modes:
      - auto: never ask, run everything
      - normal: ask for dangerous commands and sensitive file writes
      - strict: ask for all terminal commands and all file mutations
    """

    _DANGEROUS_PREFIXES = (
        "rm ", "rm\t", "rmdir ", "git push", "git reset", "git rebase",
        "git checkout -- ", "git clean", "chmod ", "chown ", "kill ",
        "killall ", "sudo ", "mkfs", "dd if=",
    )

    _SENSITIVE_PATH_PARTS = (
        ".env", "credentials", ".secret", "id_rsa", "id_ed25519",
        ".ssh/", ".aws/", "token", "password",
    )

    _MUTATING_TOOLS = ("terminal", "file_write", "file_edit")

    def __init__(
        self, callback: PermissionCallback, mode: str = "normal",
    ) -> None:
        self._callback = callback
        self._mode = mode

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value not in ("auto", "normal", "strict"):
            raise ValueError(f"Invalid permission mode: {value}")
        self._mode = value

    async def process(
        self, ctx: MiddlewareContext, call_next: Any,
    ) -> MiddlewareContext:
        if self._needs_permission(ctx):
            allowed = await self._callback(ctx.tool_name, ctx.tool_args)
            if not allowed:
                ctx.block("Denied by user")
                return ctx
        return await call_next(ctx)

    def _needs_permission(self, ctx: MiddlewareContext) -> bool:
        if self._mode == "auto":
            return False

        if self._mode == "strict":
            return ctx.tool_name in self._MUTATING_TOOLS

        # normal mode
        if ctx.tool_name == "terminal":
            cmd = ctx.tool_args.get("command", "").strip()
            for prefix in self._DANGEROUS_PREFIXES:
                if cmd.startswith(prefix) or f"&& {prefix}" in cmd or f"; {prefix}" in cmd:
                    return True
            return False

        if ctx.tool_name == "file_write":
            path = ctx.tool_args.get("path", "").lower()
            return any(part in path for part in self._SENSITIVE_PATH_PARTS)

        return False


class OutputGuardrail:
    """Redact secrets from tool output."""

    SECRET_PATTERNS = [
        r"sk-[A-Za-z0-9]{20,}",
        r"ghp_[A-Za-z0-9]{36,}",
        r"gho_[A-Za-z0-9]{36,}",
        r"AKIA[A-Z0-9]{16}",
        r"-----BEGIN [A-Z ]+ KEY-----[\s\S]*?-----END [A-Z ]+ KEY-----",
    ]

    async def process(
        self, ctx: MiddlewareContext, call_next: Any,
    ) -> MiddlewareContext:
        ctx = await call_next(ctx)
        if ctx.result and ctx.result.output:
            for pattern in self.SECRET_PATTERNS:
                ctx.result.output = re.sub(
                    pattern, "***REDACTED***", ctx.result.output,
                )
        return ctx


class CostMiddleware:
    """Track token cost and enforce budget."""

    def __init__(self, daily_limit_usd: float = 0.0) -> None:
        self._daily_limit = daily_limit_usd
        self._daily_spent: float = 0.0

    async def process(
        self, ctx: MiddlewareContext, call_next: Any,
    ) -> MiddlewareContext:
        if self._daily_limit > 0 and self._daily_spent >= self._daily_limit:
            ctx.block(f"Daily budget exhausted (${self._daily_spent:.2f})")
            return ctx
        ctx = await call_next(ctx)
        return ctx

    def record(self, cost_usd: float) -> None:
        self._daily_spent += cost_usd

    @property
    def daily_spent(self) -> float:
        return self._daily_spent


class TimingMiddleware:
    """Record tool execution duration."""

    async def process(
        self, ctx: MiddlewareContext, call_next: Any,
    ) -> MiddlewareContext:
        start = time.monotonic_ns()
        ctx = await call_next(ctx)
        elapsed_ms = (time.monotonic_ns() - start) // 1_000_000
        ctx.metadata["duration_ms"] = elapsed_ms
        return ctx
