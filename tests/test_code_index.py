"""Tests for code index — repo map, locator, and tools."""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from calamar.code_index.locator import CodeLocator
from calamar.code_index.repo_map import RepoMap, SymbolKind


PYTHON_SOURCE = """\
class UserService:
    def login(self, username: str, password: str) -> Token:
        pass

    def verify(self, token: str) -> bool:
        return True

def helper(x: int) -> str:
    return str(x)

CONSTANT = 42
"""

JAVA_SOURCE = """\
public class AuthController {
    public Token authenticate(String user, String pass) {
        return null;
    }

    private void audit(String action) {}
}

interface Authenticator {
    Token verify(String token);
}
"""

GO_SOURCE = """\
package auth

type Service struct {
    db *sql.DB
}

func NewService(db *sql.DB) *Service {
    return &Service{db: db}
}

func (s *Service) Login(user string) (*Token, error) {
    return nil, nil
}
"""

RUST_SOURCE = """\
pub struct Config {
    pub key: String,
}

impl Config {
    pub fn new(key: String) -> Self {
        Self { key }
    }

    pub fn validate(&self) -> Result<(), Error> {
        Ok(())
    }
}

pub fn init(config: &Config) -> Client {
    Client::new(config)
}
"""

TS_SOURCE = """\
export class ApiClient {
  constructor(baseUrl: string) {}
  async fetch<T>(path: string): Promise<T> {
    return null as any;
  }
}

export function createClient(url: string): ApiClient {
  return new ApiClient(url);
}

export interface Repository {
  findById(id: string): Promise<Entity>;
}
"""

JS_SOURCE = """\
class EventBus {
  constructor() {
    this.listeners = new Map();
  }
  on(event, handler) {}
  emit(event, data) {}
}

function debounce(fn, ms) {
  return () => {};
}
"""


@pytest.fixture
def multi_lang_dir():
    with tempfile.TemporaryDirectory() as d:
        files = {
            "service.py": PYTHON_SOURCE,
            "AuthController.java": JAVA_SOURCE,
            "service.go": GO_SOURCE,
            "lib.rs": RUST_SOURCE,
            "api.ts": TS_SOURCE,
            "utils.js": JS_SOURCE,
        }
        for name, content in files.items():
            with open(os.path.join(d, name), "w") as f:
                f.write(content)
        yield d


@pytest.fixture
def python_dir():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "app.py"), "w") as f:
            f.write(PYTHON_SOURCE)
        yield d


class TestRepoMapPython:
    def test_scan_finds_python(self, python_dir):
        rm = RepoMap(python_dir)
        files = rm.scan()
        assert len(files) == 1
        assert files[0].language == "python"

    def test_class_extracted(self, python_dir):
        rm = RepoMap(python_dir)
        files = rm.scan()
        symbols = files[0].symbols
        classes = [s for s in symbols if s.kind == SymbolKind.CLASS]
        assert len(classes) == 1
        assert classes[0].name == "UserService"

    def test_methods_nested(self, python_dir):
        rm = RepoMap(python_dir)
        files = rm.scan()
        cls = [s for s in files[0].symbols if s.kind == SymbolKind.CLASS][0]
        methods = cls.children
        assert len(methods) == 2
        assert methods[0].name == "login"
        assert methods[0].kind == SymbolKind.METHOD

    def test_function_extracted(self, python_dir):
        rm = RepoMap(python_dir)
        files = rm.scan()
        funcs = [s for s in files[0].symbols if s.kind == SymbolKind.FUNCTION]
        assert len(funcs) == 1
        assert funcs[0].name == "helper"

    def test_signature_includes_params(self, python_dir):
        rm = RepoMap(python_dir)
        files = rm.scan()
        cls = [s for s in files[0].symbols if s.kind == SymbolKind.CLASS][0]
        login = cls.children[0]
        assert "username: str" in login.signature
        assert "-> Token" in login.signature


class TestRepoMapMultiLang:
    def test_scans_all_languages(self, multi_lang_dir):
        rm = RepoMap(multi_lang_dir)
        files = rm.scan()
        languages = {f.language for f in files}
        assert "python" in languages
        assert "java" in languages
        assert "go" in languages
        assert "rust" in languages
        assert "typescript" in languages
        assert "javascript" in languages

    def test_java_interface(self, multi_lang_dir):
        rm = RepoMap(multi_lang_dir)
        files = rm.scan()
        java_files = [f for f in files if f.language == "java"]
        assert len(java_files) == 1
        ifaces = [s for s in java_files[0].symbols if s.kind == SymbolKind.INTERFACE]
        assert len(ifaces) == 1
        assert ifaces[0].name == "Authenticator"

    def test_go_struct_and_functions(self, multi_lang_dir):
        rm = RepoMap(multi_lang_dir)
        files = rm.scan()
        go_files = [f for f in files if f.language == "go"]
        symbols = go_files[0].symbols
        names = [s.name for s in symbols]
        assert "Service" in names
        assert "NewService" in names
        assert "Login" in names

    def test_rust_impl(self, multi_lang_dir):
        rm = RepoMap(multi_lang_dir)
        files = rm.scan()
        rs_files = [f for f in files if f.language == "rust"]
        symbols = rs_files[0].symbols
        impl_syms = [s for s in symbols if s.kind == SymbolKind.IMPL]
        assert len(impl_syms) == 1
        assert impl_syms[0].name == "Config"
        methods = impl_syms[0].children
        assert len(methods) == 2

    def test_ts_exports(self, multi_lang_dir):
        rm = RepoMap(multi_lang_dir)
        files = rm.scan()
        ts_files = [f for f in files if f.language == "typescript"]
        symbols = ts_files[0].symbols
        names = [s.name for s in symbols]
        assert "ApiClient" in names
        assert "createClient" in names
        assert "Repository" in names

    def test_js_class(self, multi_lang_dir):
        rm = RepoMap(multi_lang_dir)
        files = rm.scan()
        js_files = [f for f in files if f.language == "javascript"]
        symbols = js_files[0].symbols
        cls = [s for s in symbols if s.kind == SymbolKind.CLASS]
        assert len(cls) == 1
        assert cls[0].name == "EventBus"
        assert len(cls[0].children) == 3


class TestRepoMapRender:
    def test_render_output(self, python_dir):
        rm = RepoMap(python_dir)
        text = rm.render(max_tokens=5000)
        assert "app.py" in text
        assert "class UserService" in text
        assert "def login" in text
        assert "def helper" in text

    def test_render_truncates(self, multi_lang_dir):
        rm = RepoMap(multi_lang_dir)
        text = rm.render(max_tokens=50)
        assert len(text) < 300

    def test_render_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            rm = RepoMap(d)
            text = rm.render()
            assert text == ""


class TestRepoMapEdgeCases:
    def test_skips_large_files(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "big.py")
            with open(path, "w") as f:
                f.write("x = 1\n" * 200_000)
            rm = RepoMap(d)
            files = rm.scan()
            assert len(files) == 0

    def test_skips_node_modules(self):
        with tempfile.TemporaryDirectory() as d:
            nm = os.path.join(d, "node_modules")
            os.makedirs(nm)
            with open(os.path.join(nm, "lib.js"), "w") as f:
                f.write("function x() {}")
            with open(os.path.join(d, "app.js"), "w") as f:
                f.write("function main() {}")
            rm = RepoMap(d)
            files = rm.scan()
            assert len(files) == 1
            assert "app.js" in files[0].path

    def test_scan_specific_paths(self, python_dir):
        rm = RepoMap(python_dir)
        files = rm.scan(paths=["app.py"])
        assert len(files) == 1

    def test_unsupported_extension(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "data.csv"), "w") as f:
                f.write("a,b,c\n1,2,3\n")
            rm = RepoMap(d)
            files = rm.scan()
            assert len(files) == 0


class TestCodeLocator:
    def test_locate_by_name(self, python_dir):
        rm = RepoMap(python_dir)
        loc = CodeLocator(rm)
        result = loc.locate("UserService login")
        assert len(result.blocks) > 0
        assert any("login" in b.symbol_name for b in result.blocks)

    def test_locate_returns_source(self, python_dir):
        rm = RepoMap(python_dir)
        loc = CodeLocator(rm)
        result = loc.locate("helper")
        assert len(result.blocks) > 0
        block = result.blocks[0]
        assert "def helper" in block.source

    def test_locate_no_match(self, python_dir):
        rm = RepoMap(python_dir)
        loc = CodeLocator(rm)
        result = loc.locate("nonexistent_xyz_symbol")
        assert len(result.blocks) == 0

    def test_find_symbol_exact(self, python_dir):
        rm = RepoMap(python_dir)
        loc = CodeLocator(rm)
        blocks = loc.find_symbol("UserService")
        assert len(blocks) == 1
        assert blocks[0].symbol_name == "UserService"
        assert blocks[0].relevance == 1.0

    def test_find_symbol_partial(self, python_dir):
        rm = RepoMap(python_dir)
        loc = CodeLocator(rm)
        blocks = loc.find_symbol("User")
        assert len(blocks) >= 1
        assert blocks[0].symbol_name == "UserService"

    def test_find_symbol_with_kind(self, python_dir):
        rm = RepoMap(python_dir)
        loc = CodeLocator(rm)
        blocks = loc.find_symbol("login", kind="method")
        assert len(blocks) == 1
        assert blocks[0].symbol_name == "login"

    def test_get_file_context(self, python_dir):
        rm = RepoMap(python_dir)
        loc = CodeLocator(rm)
        ctx = loc.get_file_context("app.py")
        assert ctx is not None
        assert "UserService" in ctx

    def test_get_file_context_not_found(self, python_dir):
        rm = RepoMap(python_dir)
        loc = CodeLocator(rm)
        ctx = loc.get_file_context("nonexistent.py")
        assert ctx is None

    def test_max_results(self, multi_lang_dir):
        rm = RepoMap(multi_lang_dir)
        loc = CodeLocator(rm)
        result = loc.locate("service", max_results=2)
        assert len(result.blocks) <= 2

    def test_repo_map_excerpt(self, python_dir):
        rm = RepoMap(python_dir)
        loc = CodeLocator(rm)
        result = loc.locate("login")
        assert result.repo_map_excerpt


class TestCodeIndexTools:
    def test_code_map_tool(self, python_dir):
        from calamar.tools.code_index import CodeMapTool

        tool = CodeMapTool(working_dir=python_dir)
        result = asyncio.run(tool.execute())
        assert result.success
        assert "UserService" in result.output

    def test_code_map_tool_spec(self):
        from calamar.tools.code_index import CodeMapTool

        tool = CodeMapTool()
        assert tool.spec.name == "code_map"
        schema = tool.spec.to_openai_schema()
        assert schema["function"]["name"] == "code_map"

    def test_code_locate_tool(self, python_dir):
        from calamar.tools.code_index import CodeLocateTool

        tool = CodeLocateTool(working_dir=python_dir)
        result = asyncio.run(tool.execute(query="login"))
        assert result.success
        assert "login" in result.output

    def test_code_locate_empty_query(self):
        from calamar.tools.code_index import CodeLocateTool

        tool = CodeLocateTool()
        result = asyncio.run(tool.execute(query=""))
        assert not result.success

    def test_code_locate_no_match(self, python_dir):
        from calamar.tools.code_index import CodeLocateTool

        tool = CodeLocateTool(working_dir=python_dir)
        result = asyncio.run(tool.execute(query="xyz_nonexistent_symbol"))
        assert result.success
        assert "No code found" in result.output

    def test_code_map_empty_dir(self):
        from calamar.tools.code_index import CodeMapTool

        with tempfile.TemporaryDirectory() as d:
            tool = CodeMapTool(working_dir=d)
            result = asyncio.run(tool.execute())
            assert result.success
            assert "No supported" in result.output

    def test_code_map_subpath(self, multi_lang_dir):
        from calamar.tools.code_index import CodeMapTool

        os.makedirs(os.path.join(multi_lang_dir, "src"))
        with open(os.path.join(multi_lang_dir, "src", "mod.py"), "w") as f:
            f.write("def inner(): pass\n")

        tool = CodeMapTool(working_dir=multi_lang_dir)
        result = asyncio.run(tool.execute(path="src"))
        assert result.success
        assert "inner" in result.output

    def test_code_map_bad_path(self):
        from calamar.tools.code_index import CodeMapTool

        with tempfile.TemporaryDirectory() as d:
            tool = CodeMapTool(working_dir=d)
            result = asyncio.run(tool.execute(path="nonexistent"))
            assert not result.success
