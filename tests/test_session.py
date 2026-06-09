"""Tests for session persistence."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from calamar.context import Message
from calamar.session import SessionManager, SessionMeta, _msg_to_dict, _dict_to_msg


class TestMsgSerialization:
    def test_roundtrip_user_message(self):
        msg = Message(role="user", content="hello")
        d = _msg_to_dict(msg)
        restored = _dict_to_msg(d)
        assert restored.role == "user"
        assert restored.content == "hello"
        assert restored.name is None
        assert restored.tool_call_id is None

    def test_roundtrip_tool_message(self):
        msg = Message(
            role="tool", content="output", tool_call_id="tc_1", name="file_read",
        )
        d = _msg_to_dict(msg)
        assert d["tool_call_id"] == "tc_1"
        assert d["name"] == "file_read"
        restored = _dict_to_msg(d)
        assert restored.tool_call_id == "tc_1"
        assert restored.name == "file_read"

    def test_roundtrip_assistant_with_tool_calls(self):
        tc = [{"id": "tc_1", "type": "function", "function": {"name": "ls"}}]
        msg = Message(role="assistant", content="text", tool_calls=tc)
        d = _msg_to_dict(msg)
        assert d["tool_calls"] == tc
        restored = _dict_to_msg(d)
        assert restored.tool_calls == tc

    def test_omits_none_fields(self):
        msg = Message(role="user", content="hi")
        d = _msg_to_dict(msg)
        assert "name" not in d
        assert "tool_call_id" not in d
        assert "tool_calls" not in d


class TestSessionManager:
    def _make_mgr(self, tmp: Path) -> SessionManager:
        with patch("calamar.session._sessions_dir", return_value=tmp):
            return SessionManager(model="test-model")

    def test_save_and_load(self, tmp_path):
        with patch("calamar.session._sessions_dir", return_value=tmp_path):
            mgr = SessionManager(model="gpt-4")
            history = [
                Message(role="user", content="fix the bug"),
                Message(role="assistant", content="done"),
            ]
            mgr.save(history)

            files = list(tmp_path.glob("*.json"))
            assert len(files) == 1

            data = json.loads(files[0].read_text())
            assert data["model"] == "gpt-4"
            assert data["message_count"] == 2
            assert data["summary"] == "fix the bug"

            mgr2 = SessionManager()
            loaded = mgr2.load(mgr.session_id)
            assert len(loaded) == 2
            assert loaded[0].role == "user"
            assert loaded[0].content == "fix the bug"

    def test_list_sessions(self, tmp_path):
        with patch("calamar.session._sessions_dir", return_value=tmp_path):
            for i in range(3):
                mgr = SessionManager(model=f"model-{i}")
                mgr._created_at = float(i)
                history = [Message(role="user", content=f"task {i}")]
                mgr.save(history)

            sessions = SessionManager.list_sessions(limit=10)
            assert len(sessions) == 3
            assert sessions[0].updated_at >= sessions[1].updated_at

    def test_list_sessions_limit(self, tmp_path):
        with patch("calamar.session._sessions_dir", return_value=tmp_path):
            for i in range(5):
                mgr = SessionManager()
                mgr.save([Message(role="user", content=f"task {i}")])

            sessions = SessionManager.list_sessions(limit=2)
            assert len(sessions) == 2

    def test_delete_session(self, tmp_path):
        with patch("calamar.session._sessions_dir", return_value=tmp_path):
            mgr = SessionManager()
            mgr.save([Message(role="user", content="hello")])

            assert SessionManager.delete_session(mgr.session_id)
            assert not list(tmp_path.glob("*.json"))

    def test_delete_nonexistent(self, tmp_path):
        with patch("calamar.session._sessions_dir", return_value=tmp_path):
            assert not SessionManager.delete_session("nonexistent")

    def test_summary_skips_steering(self, tmp_path):
        with patch("calamar.session._sessions_dir", return_value=tmp_path):
            mgr = SessionManager()
            history = [
                Message(role="user", content="[steering] focus on tests"),
                Message(role="user", content="actual question"),
            ]
            mgr.save(history)

            data = json.loads(
                (tmp_path / f"{mgr.session_id}.json").read_text()
            )
            assert data["summary"] == "actual question"

    def test_summary_truncation(self, tmp_path):
        with patch("calamar.session._sessions_dir", return_value=tmp_path):
            mgr = SessionManager()
            long_msg = "x" * 200
            mgr.save([Message(role="user", content=long_msg)])

            data = json.loads(
                (tmp_path / f"{mgr.session_id}.json").read_text()
            )
            assert len(data["summary"]) == 103  # 100 + "..."
            assert data["summary"].endswith("...")
