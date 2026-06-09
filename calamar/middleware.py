"""Middleware pipeline — layered processing for every tool call."""

from __future__ import annotations

import re
import time
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
