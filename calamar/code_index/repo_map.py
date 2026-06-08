"""Repo Map — tree-sitter based symbol extraction for code understanding."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from tree_sitter import Language, Node, Parser


class SymbolKind(Enum):
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    STRUCT = "struct"
    INTERFACE = "interface"
    IMPL = "impl"


@dataclass
class Symbol:
    name: str
    kind: SymbolKind
    signature: str
    line_start: int
    line_end: int
    children: list[Symbol] = field(default_factory=list)


@dataclass
class FileSymbols:
    path: str
    language: str
    symbols: list[Symbol] = field(default_factory=list)


@dataclass
class LanguageConfig:
    name: str
    language_fn: Any
    extensions: tuple[str, ...]
    class_types: tuple[str, ...]
    function_types: tuple[str, ...]
    container_types: tuple[str, ...] = ()
    impl_types: tuple[str, ...] = ()


def _get_language_configs() -> dict[str, LanguageConfig]:
    configs: dict[str, LanguageConfig] = {}

    try:
        import tree_sitter_python as tspython
        configs["python"] = LanguageConfig(
            name="python",
            language_fn=tspython.language,
            extensions=(".py", ".pyi"),
            class_types=("class_definition",),
            function_types=("function_definition",),
        )
    except ImportError:
        pass

    try:
        import tree_sitter_javascript as tsjs
        configs["javascript"] = LanguageConfig(
            name="javascript",
            language_fn=tsjs.language,
            extensions=(".js", ".jsx", ".mjs", ".cjs"),
            class_types=("class_declaration",),
            function_types=("function_declaration", "method_definition"),
            container_types=("class_body",),
        )
    except ImportError:
        pass

    try:
        import tree_sitter_typescript as tsts
        configs["typescript"] = LanguageConfig(
            name="typescript",
            language_fn=tsts.language_typescript,
            extensions=(".ts",),
            class_types=("class_declaration", "interface_declaration"),
            function_types=(
                "function_declaration",
                "method_definition",
                "function_signature",
                "method_signature",
            ),
            container_types=("class_body", "interface_body"),
        )
        configs["tsx"] = LanguageConfig(
            name="tsx",
            language_fn=tsts.language_tsx,
            extensions=(".tsx",),
            class_types=("class_declaration", "interface_declaration"),
            function_types=(
                "function_declaration",
                "method_definition",
                "function_signature",
                "method_signature",
            ),
            container_types=("class_body", "interface_body"),
        )
    except ImportError:
        pass

    try:
        import tree_sitter_java as tsjava
        configs["java"] = LanguageConfig(
            name="java",
            language_fn=tsjava.language,
            extensions=(".java",),
            class_types=("class_declaration", "interface_declaration"),
            function_types=("method_declaration", "constructor_declaration"),
            container_types=("class_body",),
        )
    except ImportError:
        pass

    try:
        import tree_sitter_go as tsgo
        configs["go"] = LanguageConfig(
            name="go",
            language_fn=tsgo.language,
            extensions=(".go",),
            class_types=("type_declaration",),
            function_types=("function_declaration", "method_declaration"),
        )
    except ImportError:
        pass

    try:
        import tree_sitter_rust as tsrust
        configs["rust"] = LanguageConfig(
            name="rust",
            language_fn=tsrust.language,
            extensions=(".rs",),
            class_types=("struct_item", "enum_item", "trait_item"),
            function_types=("function_item",),
            impl_types=("impl_item",),
        )
    except ImportError:
        pass

    return configs


SKIP_DIRS = {
    ".git", ".hg", ".svn",
    "node_modules", "__pycache__", ".venv", "venv",
    ".tox", ".mypy_cache", ".ruff_cache", ".pytest_cache",
    "dist", "build", "target", ".next", ".nuxt",
    "vendor", "third_party",
}

MAX_FILE_SIZE = 500_000


class RepoMap:
    """Builds a compact symbol map of a code repository using tree-sitter."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        self._configs = _get_language_configs()
        self._parsers: dict[str, Parser] = {}
        self._ext_to_lang: dict[str, str] = {}

        for lang_name, config in self._configs.items():
            for ext in config.extensions:
                self._ext_to_lang[ext] = lang_name

    @property
    def supported_extensions(self) -> set[str]:
        return set(self._ext_to_lang.keys())

    def _get_parser(self, lang_name: str) -> Parser:
        if lang_name not in self._parsers:
            config = self._configs[lang_name]
            lang = Language(config.language_fn())
            self._parsers[lang_name] = Parser(lang)
        return self._parsers[lang_name]

    def scan(self, paths: list[str] | None = None) -> list[FileSymbols]:
        """Scan the repository and extract symbols from all supported files."""
        if paths:
            files = [Path(p) if Path(p).is_absolute() else self._root / p for p in paths]
        else:
            files = list(self._collect_files())

        results = []
        for file_path in sorted(files):
            fs = self._parse_file(file_path)
            if fs and fs.symbols:
                results.append(fs)
        return results

    def render(
        self,
        file_symbols: list[FileSymbols] | None = None,
        max_tokens: int = 2000,
    ) -> str:
        """Render a compact text map of the repository symbols.

        Args:
            file_symbols: Pre-scanned symbols; scans the repo if None.
            max_tokens: Approximate token budget (1 token ~ 4 chars).
        """
        if file_symbols is None:
            file_symbols = self.scan()

        lines: list[str] = []
        char_budget = max_tokens * 4

        for fs in file_symbols:
            rel = self._relative_path(fs.path)
            file_lines = self._render_file(rel, fs.symbols, indent=0)
            lines.extend(file_lines)

        text = "\n".join(lines)
        if len(text) > char_budget:
            text = self._truncate(file_symbols, char_budget)
        return text

    def _collect_files(self) -> list[Path]:
        files = []
        for dirpath, dirnames, filenames in os.walk(self._root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fname in filenames:
                fpath = Path(dirpath) / fname
                ext = fpath.suffix
                if ext in self._ext_to_lang:
                    try:
                        if fpath.stat().st_size <= MAX_FILE_SIZE:
                            files.append(fpath)
                    except OSError:
                        pass
        return files

    def _parse_file(self, file_path: Path) -> FileSymbols | None:
        ext = file_path.suffix
        lang_name = self._ext_to_lang.get(ext)
        if not lang_name:
            return None

        try:
            source = file_path.read_bytes()
        except OSError:
            return None

        config = self._configs[lang_name]
        parser = self._get_parser(lang_name)
        tree = parser.parse(source)

        symbols = self._extract_symbols(tree.root_node, source, config)
        return FileSymbols(
            path=str(file_path),
            language=lang_name,
            symbols=symbols,
        )

    def _extract_symbols(
        self, node: Node, source: bytes, config: LanguageConfig,
    ) -> list[Symbol]:
        symbols: list[Symbol] = []
        for child in node.children:
            self._collect_symbol(child, source, config, symbols)
        return symbols

    def _collect_symbol(
        self,
        node: Node,
        source: bytes,
        config: LanguageConfig,
        out: list[Symbol],
    ) -> None:
        sym = self._node_to_symbol(node, source, config)
        if sym:
            out.append(sym)
            return

        if node.type in ("export_statement", "export_default_declaration"):
            for child in node.children:
                self._collect_symbol(child, source, config, out)
            return

        if node.type == "decorated_definition":
            for child in node.children:
                sym = self._node_to_symbol(child, source, config)
                if sym:
                    out.append(sym)
                    return

        if node.type in config.impl_types:
            impl_sym = self._extract_impl(node, source, config)
            if impl_sym:
                out.append(impl_sym)

    def _node_to_symbol(
        self, node: Node, source: bytes, config: LanguageConfig,
    ) -> Symbol | None:
        if node.type in config.class_types:
            return self._extract_class(node, source, config)
        if node.type in config.function_types:
            return self._extract_function(node, source, config)
        return None

    def _extract_class(
        self, node: Node, source: bytes, config: LanguageConfig,
    ) -> Symbol:
        name = self._get_name(node, config)
        sig = self._get_signature_line(node, source)

        children: list[Symbol] = []
        body = node.child_by_field_name("body")
        if body:
            for child in body.children:
                sym = self._node_to_symbol(child, source, config)
                if sym:
                    if sym.kind == SymbolKind.FUNCTION:
                        sym = Symbol(
                            name=sym.name,
                            kind=SymbolKind.METHOD,
                            signature=sym.signature,
                            line_start=sym.line_start,
                            line_end=sym.line_end,
                        )
                    children.append(sym)
                elif child.type == "decorated_definition":
                    for sub in child.children:
                        sym = self._node_to_symbol(sub, source, config)
                        if sym:
                            if sym.kind == SymbolKind.FUNCTION:
                                sym = Symbol(
                                    name=sym.name,
                                    kind=SymbolKind.METHOD,
                                    signature=sym.signature,
                                    line_start=sym.line_start,
                                    line_end=sym.line_end,
                                )
                            children.append(sym)
                            break

        kind = SymbolKind.INTERFACE if "interface" in node.type else SymbolKind.CLASS
        if "struct" in node.type:
            kind = SymbolKind.STRUCT

        return Symbol(
            name=name,
            kind=kind,
            signature=sig,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            children=children,
        )

    def _extract_function(
        self, node: Node, source: bytes, config: LanguageConfig,
    ) -> Symbol:
        name = self._get_name(node, config)
        sig = self._get_signature_line(node, source)
        return Symbol(
            name=name,
            kind=SymbolKind.FUNCTION,
            signature=sig,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
        )

    def _extract_impl(
        self, node: Node, source: bytes, config: LanguageConfig,
    ) -> Symbol | None:
        type_node = None
        for child in node.children:
            if child.type == "type_identifier":
                type_node = child
                break

        if not type_node:
            return None

        name = type_node.text.decode()
        children: list[Symbol] = []
        body = node.child_by_field_name("body")
        if not body:
            for child in node.children:
                if child.type == "declaration_list":
                    body = child
                    break

        if body:
            for child in body.children:
                if child.type in config.function_types:
                    sym = self._extract_function(child, source, config)
                    sym = Symbol(
                        name=sym.name,
                        kind=SymbolKind.METHOD,
                        signature=sym.signature,
                        line_start=sym.line_start,
                        line_end=sym.line_end,
                    )
                    children.append(sym)

        return Symbol(
            name=name,
            kind=SymbolKind.IMPL,
            signature=f"impl {name}",
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            children=children,
        )

    def _get_name(self, node: Node, config: LanguageConfig) -> str:
        name_node = node.child_by_field_name("name")
        if name_node:
            return name_node.text.decode()

        for child in node.children:
            if child.type in ("identifier", "type_identifier",
                              "field_identifier", "property_identifier",
                              "type_spec"):
                if child.type == "type_spec":
                    for sub in child.children:
                        if sub.type == "type_identifier":
                            return sub.text.decode()
                return child.text.decode()
        return "?"

    def _get_signature_line(self, node: Node, source: bytes) -> str:
        body = node.child_by_field_name("body")
        if body:
            sig_bytes = source[node.start_byte:body.start_byte]
            sig = sig_bytes.decode(errors="replace").strip()
            sig = sig.rstrip("{:").strip()
            return sig

        for child in node.children:
            if child.type in ("block", "class_body", "statement_block",
                              "declaration_list", "field_declaration_list"):
                sig_bytes = source[node.start_byte:child.start_byte]
                sig = sig_bytes.decode(errors="replace").strip()
                sig = sig.rstrip("{:").strip()
                return sig

        line_start = node.start_point[0]
        line_end = node.start_point[0]
        text = source.decode(errors="replace").split("\n")
        if line_start < len(text):
            sig = text[line_start].strip().rstrip("{:;").strip()
            return sig
        return node.text.decode(errors="replace").split("\n")[0].strip()

    def _relative_path(self, path: str) -> str:
        try:
            return str(Path(path).relative_to(self._root))
        except ValueError:
            return path

    def _render_file(
        self, rel_path: str, symbols: list[Symbol], indent: int,
    ) -> list[str]:
        lines = [f"{rel_path}"]
        for sym in symbols:
            lines.extend(self._render_symbol(sym, indent=1))
        return lines

    def _render_symbol(self, sym: Symbol, indent: int) -> list[str]:
        prefix = "  " * indent
        lines = [f"{prefix}{sym.signature}"]
        for child in sym.children:
            lines.extend(self._render_symbol(child, indent + 1))
        return lines

    def _truncate(self, file_symbols: list[FileSymbols], budget: int) -> str:
        lines: list[str] = []
        used = 0
        for fs in file_symbols:
            rel = self._relative_path(fs.path)
            header = rel
            if used + len(header) > budget:
                break
            lines.append(header)
            used += len(header) + 1

            for sym in fs.symbols:
                sym_lines = self._render_symbol(sym, indent=1)
                sym_text = "\n".join(sym_lines)
                if used + len(sym_text) > budget:
                    lines.append("  ...")
                    return "\n".join(lines)
                lines.extend(sym_lines)
                used += len(sym_text) + 1

        return "\n".join(lines)
