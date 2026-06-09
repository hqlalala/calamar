"""Session persistence — save / resume conversation history."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from calamar.context import Message


@dataclass
class SessionMeta:
    session_id: str
    model: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    message_count: int = 0
    summary: str = ""


def _sessions_dir() -> Path:
    d = Path.home() / ".calamar" / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _msg_to_dict(msg: Message) -> dict[str, Any]:
    d: dict[str, Any] = {"role": msg.role, "content": msg.content}
    if msg.name:
        d["name"] = msg.name
    if msg.tool_call_id:
        d["tool_call_id"] = msg.tool_call_id
    if msg.tool_calls:
        d["tool_calls"] = msg.tool_calls
    return d


def _dict_to_msg(d: dict[str, Any]) -> Message:
    return Message(
        role=d["role"],
        content=d.get("content", ""),
        name=d.get("name"),
        tool_call_id=d.get("tool_call_id"),
        tool_calls=d.get("tool_calls"),
    )


class SessionManager:
    """Manages conversation sessions on disk."""

    def __init__(self, model: str = "") -> None:
        self._session_id = uuid.uuid4().hex[:12]
        self._model = model
        self._created_at = time.time()

    @property
    def session_id(self) -> str:
        return self._session_id

    def save(self, history: list[Message]) -> Path:
        """Save current session to disk."""
        now = time.time()
        summary = self._extract_summary(history)
        data = {
            "session_id": self._session_id,
            "model": self._model,
            "created_at": self._created_at,
            "updated_at": now,
            "message_count": len(history),
            "summary": summary,
            "messages": [_msg_to_dict(m) for m in history],
        }
        path = _sessions_dir() / f"{self._session_id}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        return path

    def load(self, session_id: str) -> list[Message]:
        """Load a session's messages from disk."""
        path = _sessions_dir() / f"{session_id}.json"
        with open(path) as f:
            data = json.load(f)
        self._session_id = data["session_id"]
        self._model = data.get("model", "")
        self._created_at = data.get("created_at", time.time())
        return [_dict_to_msg(m) for m in data.get("messages", [])]

    @staticmethod
    def list_sessions(limit: int = 10) -> list[SessionMeta]:
        """List recent sessions, newest first."""
        sessions_dir = _sessions_dir()
        entries: list[SessionMeta] = []
        for path in sessions_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                entries.append(SessionMeta(
                    session_id=data["session_id"],
                    model=data.get("model", ""),
                    created_at=data.get("created_at", 0),
                    updated_at=data.get("updated_at", 0),
                    message_count=data.get("message_count", 0),
                    summary=data.get("summary", ""),
                ))
            except (json.JSONDecodeError, OSError, KeyError):
                continue
        entries.sort(key=lambda s: s.updated_at, reverse=True)
        return entries[:limit]

    @staticmethod
    def delete_session(session_id: str) -> bool:
        """Delete a session file."""
        path = _sessions_dir() / f"{session_id}.json"
        try:
            path.unlink()
            return True
        except OSError:
            return False

    @staticmethod
    def _extract_summary(history: list[Message]) -> str:
        """Extract first user message as session summary."""
        for msg in history:
            if msg.role == "user" and not msg.content.startswith("[steering]"):
                text = msg.content.strip().replace("\n", " ")
                return text[:100] + "..." if len(text) > 100 else text
        return ""
