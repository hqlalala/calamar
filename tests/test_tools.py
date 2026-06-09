"""Tests for built-in tools."""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from calamar.tools.defaults import create_default_registry
from calamar.tools.file import FileEditTool, FileReadTool, FileWriteTool
from calamar.tools.search import ListDirectoryTool, SearchCodeTool
from calamar.tools.terminal import TerminalTool


class TestTerminalTool:
    def test_spec(self):
        tool = TerminalTool()
        assert tool.spec.name == "terminal"
        schema = tool.spec.to_openai_schema()
        assert schema["function"]["name"] == "terminal"

    def test_echo(self):
        tool = TerminalTool()
        result = asyncio.run(tool.execute(command="echo hello"))
        assert result.success
        assert "hello" in result.output

    def test_empty_command(self):
        tool = TerminalTool()
        result = asyncio.run(tool.execute(command=""))
        assert not result.success

    def test_timeout(self):
        tool = TerminalTool()
        result = asyncio.run(tool.execute(command="sleep 10", timeout=1))
        assert not result.success
        assert "timed out" in result.error

    def test_nonzero_exit(self):
        tool = TerminalTool()
        result = asyncio.run(tool.execute(command="false"))
        assert result.error is not None
        assert result.metadata["exit_code"] != 0

    def test_working_dir(self):
        tool = TerminalTool(working_dir="/tmp")
        result = asyncio.run(tool.execute(command="pwd"))
        assert result.success
        assert "/tmp" in result.output or "/private/tmp" in result.output


class TestFileReadTool:
    def test_read_file(self):
        tool = FileReadTool()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("line1\nline2\nline3\n")
            path = f.name
        try:
            result = asyncio.run(tool.execute(path=path))
            assert result.success
            assert "line1" in result.output
            assert "line3" in result.output
            assert result.metadata["total_lines"] == 3
        finally:
            os.unlink(path)

    def test_read_with_offset(self):
        tool = FileReadTool()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("a\nb\nc\nd\ne\n")
            path = f.name
        try:
            result = asyncio.run(tool.execute(path=path, offset=2, limit=2))
            assert result.success
            assert "c" in result.output
            assert "d" in result.output
        finally:
            os.unlink(path)

    def test_not_found(self):
        tool = FileReadTool()
        result = asyncio.run(tool.execute(path="/nonexistent/file.txt"))
        assert not result.success

    def test_directory(self):
        tool = FileReadTool()
        result = asyncio.run(tool.execute(path="/tmp"))
        assert not result.success
        assert "directory" in result.error.lower()


class TestFileWriteTool:
    def test_write_new_file(self):
        tool = FileWriteTool()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.txt")
            result = asyncio.run(tool.execute(path=path, content="hello world\n"))
            assert result.success
            with open(path) as f:
                assert f.read() == "hello world\n"

    def test_create_parent_dirs(self):
        tool = FileWriteTool()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "dir", "test.txt")
            result = asyncio.run(tool.execute(path=path, content="nested\n"))
            assert result.success
            assert os.path.exists(path)


class TestFileEditTool:
    def test_edit_replaces(self):
        tool = FileEditTool()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def hello():\n    return 'hello'\n")
            path = f.name
        try:
            result = asyncio.run(tool.execute(
                path=path,
                old_string="return 'hello'",
                new_string="return 'world'",
            ))
            assert result.success
            with open(path) as f:
                assert "return 'world'" in f.read()
        finally:
            os.unlink(path)

    def test_not_found_string(self):
        tool = FileEditTool()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("some content\n")
            path = f.name
        try:
            result = asyncio.run(tool.execute(
                path=path, old_string="nonexistent", new_string="replacement",
            ))
            assert not result.success
        finally:
            os.unlink(path)

    def test_ambiguous_match(self):
        tool = FileEditTool()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("foo\nfoo\n")
            path = f.name
        try:
            result = asyncio.run(tool.execute(
                path=path, old_string="foo", new_string="bar",
            ))
            assert not result.success
            assert "2 times" in result.error
        finally:
            os.unlink(path)


class TestSearchCodeTool:
    def test_search_pattern(self):
        tool = SearchCodeTool()
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "test.py"), "w") as f:
                f.write("def hello():\n    pass\n")
            tool_in_dir = SearchCodeTool(working_dir=d)
            result = asyncio.run(tool_in_dir.execute(pattern="hello", path="."))
            assert result.success
            assert "hello" in result.output

    def test_no_match(self):
        tool = SearchCodeTool()
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "test.py"), "w") as f:
                f.write("def foo():\n    pass\n")
            tool_in_dir = SearchCodeTool(working_dir=d)
            result = asyncio.run(tool_in_dir.execute(
                pattern="nonexistent_symbol_xyz", path=".",
            ))
            assert "No matches" in result.output


class TestListDirectoryTool:
    def test_list_dir(self):
        tool = ListDirectoryTool(working_dir="/tmp")
        result = asyncio.run(tool.execute(path="."))
        assert result.success

    def test_list_with_pattern(self):
        tool = ListDirectoryTool()
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "a.py"), "w").close()
            open(os.path.join(d, "b.txt"), "w").close()
            tool_in_dir = ListDirectoryTool(working_dir=d)
            result = asyncio.run(tool_in_dir.execute(path=".", pattern="*.py"))
            assert result.success
            assert "a.py" in result.output


class TestDefaultRegistry:
    def test_creates_all_tools(self):
        registry = create_default_registry()
        assert len(registry) >= 7
        assert "terminal" in registry
        assert "file_read" in registry
        assert "file_write" in registry
        assert "file_edit" in registry
        assert "search_code" in registry
        assert "list_directory" in registry
        assert "web_fetch" in registry

    def test_code_index_tools_when_available(self):
        try:
            import tree_sitter  # noqa: F401
            has_ts = True
        except ImportError:
            has_ts = False

        registry = create_default_registry()
        if has_ts:
            assert "code_map" in registry
            assert "code_locate" in registry
            assert len(registry) == 9
        else:
            assert "code_map" not in registry
            assert len(registry) == 7

    def test_openai_schemas(self):
        registry = create_default_registry()
        schemas = registry.to_openai_tools()
        assert len(schemas) >= 7
        names = {s["function"]["name"] for s in schemas}
        assert "terminal" in names
