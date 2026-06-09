"""Ducky (Aone Copilot) provider — native + prompt-based tool calling."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx

from calamar.config import Config
from calamar.events import TokenUsage
from calamar.providers import (
    CompletionResponse,
    StreamDelta,
    ToolCallData,
    make_tool_call_id,
)

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)

_CHUNK_TIMEOUT = 30.0


class _NativeUnsupported(Exception):
    """Raised when /v1/chat/completions is not available."""


JETBRAINS_BASE = Path.home() / "Library" / "Application Support" / "JetBrains"
LINUX_JETBRAINS_BASE = Path.home() / ".config" / "JetBrains"


class DuckyProvider:
    """Provider for Alibaba Aone Copilot (Ducky) inference gateway.

    Supports two tool-calling modes:
    - Native: OpenAI-compatible /v1/chat/completions with function calling
    - Prompt-based: Custom /v1/chat with <tool_call> XML tags in the prompt

    On first request with tools, tries native mode. If the endpoint returns
    404/405, falls back to prompt-based and caches the result for the session.
    """

    def __init__(self, config: Config) -> None:
        token = getattr(config, "ducky_token", "") or ""
        if not token:
            token = _resolve_ducky_token()
        if not token:
            raise ValueError(
                "Ducky token not found. Set AONE_TOKEN env var, "
                "or login via IDEA Aone Copilot plugin first."
            )
        self._token = token
        self._base_url = (config.base_url or "https://ducky.code.alibaba-inc.com").rstrip("/")
        self._native_supported: bool | None = None

    def _auth_headers(self, model: str = "") -> dict[str, str]:
        encoded = base64.b64encode(self._token.encode("utf-8")).decode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {encoded}",
            "Accept": "text/event-stream",
            "x-plugin-version": "2.11.8-calamar",
            "x-client-type": "calamar",
        }
        if model:
            headers["X-Model-Name"] = model
        return headers

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> CompletionResponse:
        accumulated_text = ""
        tool_calls: list[ToolCallData] = []
        usage = TokenUsage(model=model)

        async for delta in self.stream(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            if delta.text:
                accumulated_text += delta.text
            if delta.tool_calls:
                tool_calls = delta.tool_calls
            if delta.usage:
                usage = delta.usage

        finish_reason = "tool_calls" if tool_calls else "stop"
        return CompletionResponse(
            text=accumulated_text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
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
        if tools and self._native_supported is not False:
            try:
                async for delta in self._stream_native(
                    model, messages, tools, temperature, max_tokens,
                ):
                    yield delta
                self._native_supported = True
                return
            except _NativeUnsupported:
                self._native_supported = False

        async for delta in self._stream_prompt_based(
            model, messages, tools, temperature, max_tokens,
        ):
            yield delta

    # ── Native mode: OpenAI-compatible /v1/chat/completions ──────────

    async def _stream_native(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[StreamDelta]:
        """Stream via OpenAI-compatible endpoint with native tool calling."""
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools

        tc_id_map: dict[int, str] = {}
        tc_name_map: dict[int, str] = {}
        tc_args_map: dict[int, str] = {}
        finish_reason: str | None = None

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=15.0),
        ) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/v1/chat/completions",
                headers=self._auth_headers(model),
                json=body,
            ) as resp:
                if resp.status_code in (404, 405):
                    raise _NativeUnsupported()
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    raise RuntimeError(
                        f"Ducky API error HTTP {resp.status_code}: "
                        f"{error_body.decode('utf-8', errors='replace')[:500]}"
                    )

                line_iter = resp.aiter_lines()
                while True:
                    try:
                        line = await asyncio.wait_for(
                            anext(line_iter), timeout=_CHUNK_TIMEOUT,
                        )
                    except (StopAsyncIteration, TimeoutError):
                        break

                    line = line.strip()
                    if not line:
                        continue
                    if line == "data: [DONE]" or line == "[DONE]":
                        break
                    if not line.startswith("data:"):
                        continue

                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        break

                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        usage_data = chunk.get("usage")
                        if usage_data:
                            yield StreamDelta(usage=TokenUsage(
                                prompt_tokens=usage_data.get("prompt_tokens", 0),
                                completion_tokens=usage_data.get("completion_tokens", 0),
                                total_tokens=usage_data.get("total_tokens", 0),
                                model=model,
                            ))
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

                    content = delta.get("content")
                    if content:
                        yield StreamDelta(text=content)

                    for tc_delta in delta.get("tool_calls", []):
                        idx = tc_delta.get("index", 0)
                        if tc_delta.get("id"):
                            tc_id_map[idx] = tc_delta["id"]
                        func = tc_delta.get("function", {})
                        if func.get("name"):
                            tc_name_map[idx] = func["name"]
                        if func.get("arguments"):
                            tc_args_map[idx] = (
                                tc_args_map.get(idx, "") + func["arguments"]
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
        yield StreamDelta(
            tool_calls=tool_calls or None,
            finish_reason=fr,
            usage=TokenUsage(model=model),
        )

    # ── Prompt-based mode: custom /v1/chat with <tool_call> tags ─────

    async def _stream_prompt_based(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[StreamDelta]:
        """Stream via Ducky custom endpoint with prompt-based tool calling."""
        chat_messages = self._convert_messages(messages, tools)
        full_text = ""
        in_tool_call = False

        async for delta in self._stream_sse(chat_messages, model):
            full_text += delta

            if not in_tool_call:
                tc_start = delta.find("<tool_call>")
                if tc_start != -1:
                    visible = delta[:tc_start]
                    if visible:
                        yield StreamDelta(text=visible)
                    in_tool_call = True
                else:
                    yield StreamDelta(text=delta)

        tool_calls = self._parse_tool_calls(full_text)
        finish_reason = "tool_calls" if tool_calls else "stop"
        yield StreamDelta(
            tool_calls=tool_calls or None,
            finish_reason=finish_reason,
            usage=TokenUsage(model=model),
        )

    def _convert_messages(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, str]]:
        """Convert OpenAI-format messages to Ducky chatMessage format."""
        result: list[dict[str, str]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system" and tools:
                content = content + "\n\n" + _build_tools_prompt(tools)

            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                name = msg.get("name", "")
                content = (
                    f'<tool_result tool_call_id="{tool_call_id}" name="{name}">'
                    f"\n{content}\n</tool_result>"
                )
                role = "user"

            if role == "assistant" and msg.get("tool_calls"):
                tc_text = _reconstruct_tool_call_text(msg["tool_calls"])
                content = (content + "\n\n" + tc_text).strip()

            result.append({"role": role, "content": content})

        return result

    async def _stream_sse(
        self,
        chat_messages: list[dict[str, str]],
        model: str,
    ) -> AsyncIterator[str]:
        """Yield text deltas from the Ducky SSE stream."""
        headers = self._auth_headers(model)
        body = {"chatMessage": chat_messages, "needAppend": True}

        prev_len = 0
        tool_call_end_tag = "</tool_call>"

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=15.0),
        ) as client:
            async with client.stream(
                "POST", f"{self._base_url}/v1/chat",
                headers=headers, json=body,
            ) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    raise RuntimeError(
                        f"Ducky API error HTTP {resp.status_code}: "
                        f"{error_body.decode('utf-8', errors='replace')[:500]}"
                    )

                line_iter = resp.aiter_lines()
                while True:
                    try:
                        line = await asyncio.wait_for(
                            anext(line_iter), timeout=_CHUNK_TIMEOUT,
                        )
                    except (StopAsyncIteration, TimeoutError):
                        break

                    line = line.strip()
                    if not line:
                        continue

                    if line == "[DONE]" or line == "data: [DONE]":
                        break

                    data = line
                    if data.startswith("data:"):
                        data = data[5:].strip()
                        if not data or data == "[DONE]":
                            break

                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    content = obj.get("content")
                    if content is None or len(content) <= prev_len:
                        continue

                    delta = content[prev_len:]
                    prev_len = len(content)
                    yield delta

                    if tool_call_end_tag in content:
                        break

    # ── Helpers ───────────────────────────────────────────────────────

    def _parse_tool_calls(self, text: str) -> list[ToolCallData]:
        calls: list[ToolCallData] = []
        for match in _TOOL_CALL_RE.finditer(text):
            try:
                raw = json.loads(match.group(1))
                if "name" in raw:
                    calls.append(ToolCallData(
                        id=make_tool_call_id(),
                        name=raw["name"],
                        arguments=raw.get("arguments", {}),
                    ))
            except json.JSONDecodeError:
                continue
        return calls

    async def list_models(self) -> list[str]:
        headers = self._auth_headers()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self._base_url}/v1/models", headers=headers,
                )
                if resp.status_code != 200:
                    return []
                data = resp.json()
                if isinstance(data, dict):
                    data = data.get("data", data.get("models", []))
                if not isinstance(data, list):
                    return []
                results = []
                for m in data:
                    if not isinstance(m, dict):
                        continue
                    mid = m.get("id", "")
                    display = m.get("display", "")
                    if not mid:
                        continue
                    label = f"{mid} ({display})" if display and display != mid else mid
                    results.append(label)
                return sorted(results)
        except Exception:
            return []


def _build_tools_prompt(tools: list[dict[str, Any]]) -> str:
    """Convert OpenAI tool schemas to a text prompt for the model."""
    lines = [
        "## Available Tools",
        'Call a tool using: <tool_call>{"name":"tool_name","arguments":{...}}</tool_call>',
        "",
        "RULES:",
        "- Output ONLY ONE <tool_call> per response.",
        "- STOP writing immediately after </tool_call>. Do not write anything after it.",
        "- NEVER predict or write <tool_result> yourself. Wait for the actual result.",
        "- When you need to perform an action, use a tool immediately. Do NOT describe what you plan to do — just do it.",
        "- Keep explanations brief. Prefer action over narration.",
        "",
    ]
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        lines.append(f"### {name}")
        lines.append(desc)
        props = params.get("properties", {})
        required = params.get("required", [])
        if props:
            param_parts = []
            for pname, pdef in props.items():
                req = "(required)" if pname in required else "(optional)"
                ptype = pdef.get("type", "string")
                pdesc = pdef.get("description", "")
                param_parts.append(f"  - {pname} ({ptype}, {req}): {pdesc}")
            lines.append("Parameters:")
            lines.extend(param_parts)
        lines.append("")

    return "\n".join(lines)


def _reconstruct_tool_call_text(tool_calls: list[dict[str, Any]]) -> str:
    """Reconstruct <tool_call> XML from OpenAI-format tool_calls dicts."""
    parts = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        args_str = func.get("arguments", "{}")
        if isinstance(args_str, str):
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}
        else:
            args = args_str
        call_obj = {"name": name, "arguments": args}
        parts.append(f"<tool_call>{json.dumps(call_obj)}</tool_call>")
    return "\n".join(parts)


def _resolve_ducky_token() -> str:
    """Auto-detect Ducky token from env var or IDEA plugin config."""
    token = os.environ.get("AONE_TOKEN", "").strip()
    if token:
        return token

    config_file = Path.home() / ".aone-cli" / "config.json"
    if config_file.exists():
        try:
            data = json.loads(config_file.read_text("utf-8"))
            token = data.get("user_token", "").strip()
            if token:
                return token
        except Exception:
            pass

    for jb_base in (JETBRAINS_BASE, LINUX_JETBRAINS_BASE):
        if not jb_base.exists():
            continue
        candidates = []
        for d in jb_base.iterdir():
            settings = d / "options" / "AoneCopilotSettings.xml"
            if settings.exists():
                candidates.append(settings)
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for settings_file in candidates:
            token = _parse_idea_token(settings_file)
            if token:
                return token

    return ""


def _parse_idea_token(path: Path) -> str:
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        component = root.find(".//component[@name='AoneCopilotSettings']")
        if component is None:
            return ""
        for option in component.findall("option"):
            if option.get("name") == "userToken":
                return option.get("value", "")
    except Exception:
        pass
    return ""
