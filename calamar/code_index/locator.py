"""Code Locator — hierarchical code search from files to symbols to blocks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from calamar.code_index.repo_map import FileSymbols, RepoMap, Symbol


@dataclass
class CodeBlock:
    """A located code block with its surrounding context."""

    file_path: str
    symbol_name: str
    line_start: int
    line_end: int
    source: str
    relevance: float = 0.0


@dataclass
class LocateResult:
    """Result of a code locate operation."""

    query: str
    blocks: list[CodeBlock] = field(default_factory=list)
    repo_map_excerpt: str = ""


class CodeLocator:
    """Hierarchical code locator: repo → files → symbols → code blocks.

    Uses the repo map for coarse filtering, then reads source files to
    extract precise code blocks matching the query.
    """

    def __init__(self, repo_map: RepoMap) -> None:
        self._repo_map = repo_map
        self._file_symbols: list[FileSymbols] | None = None

    def _ensure_scanned(self) -> list[FileSymbols]:
        if self._file_symbols is None:
            self._file_symbols = self._repo_map.scan()
        return self._file_symbols

    def locate(
        self,
        query: str,
        max_results: int = 5,
        context_lines: int = 3,
    ) -> LocateResult:
        """Locate code blocks relevant to the query.

        Args:
            query: Natural language or symbol name query.
            max_results: Maximum number of code blocks to return.
            context_lines: Lines of context above/below the symbol.
        """
        all_files = self._ensure_scanned()
        keywords = self._extract_keywords(query)

        scored: list[tuple[float, FileSymbols, Symbol]] = []
        for fs in all_files:
            file_score = self._score_file(fs, keywords)
            for sym in self._flatten_symbols(fs.symbols):
                sym_score = self._score_symbol(sym, keywords)
                total = file_score * 0.3 + sym_score * 0.7
                if total > 0:
                    scored.append((total, fs, sym))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max_results]

        blocks: list[CodeBlock] = []
        for score, fs, sym in top:
            block = self._extract_block(fs.path, sym, context_lines)
            if block:
                block.relevance = score
                blocks.append(block)

        relevant_paths = {fs.path for _, fs, _ in top}
        excerpt = self._repo_map.render(
            [fs for fs in all_files if fs.path in relevant_paths],
            max_tokens=1000,
        )

        return LocateResult(query=query, blocks=blocks, repo_map_excerpt=excerpt)

    def find_symbol(self, name: str, kind: str = "") -> list[CodeBlock]:
        """Find a symbol by exact or partial name match."""
        all_files = self._ensure_scanned()
        results: list[CodeBlock] = []
        name_lower = name.lower()

        for fs in all_files:
            for sym in self._flatten_symbols(fs.symbols):
                if name_lower in sym.name.lower():
                    if kind and sym.kind.value != kind:
                        continue
                    block = self._extract_block(fs.path, sym, context_lines=2)
                    if block:
                        block.relevance = (
                            1.0 if sym.name.lower() == name_lower else 0.5
                        )
                        results.append(block)

        results.sort(key=lambda b: b.relevance, reverse=True)
        return results

    def get_file_context(self, file_path: str) -> str | None:
        """Get the repo map excerpt for a specific file."""
        all_files = self._ensure_scanned()
        resolved = str(Path(file_path).resolve())

        matching = [fs for fs in all_files if fs.path == resolved]
        if not matching:
            rel = file_path
            matching = [fs for fs in all_files if fs.path.endswith(rel)]

        if matching:
            return self._repo_map.render(matching, max_tokens=2000)
        return None

    def _extract_keywords(self, query: str) -> list[str]:
        words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", query)
        stop = {
            "the", "a", "an", "in", "on", "at", "to", "for", "of",
            "is", "it", "and", "or", "not", "this", "that", "with",
            "from", "by", "as", "be", "has", "have", "had", "do",
            "does", "did", "will", "would", "could", "should", "can",
            "find", "show", "get", "what", "where", "how", "which",
            "all", "code", "function", "method", "class", "file",
        }
        return [w.lower() for w in words if w.lower() not in stop and len(w) > 1]

    def _score_file(self, fs: FileSymbols, keywords: list[str]) -> float:
        path_lower = fs.path.lower()
        score = 0.0
        for kw in keywords:
            if kw in path_lower:
                score += 1.0
        return min(score, 3.0)

    def _score_symbol(self, sym: Symbol, keywords: list[str]) -> float:
        name_lower = sym.name.lower()
        sig_lower = sym.signature.lower()
        score = 0.0

        for kw in keywords:
            if kw == name_lower:
                score += 3.0
            elif kw in name_lower:
                score += 2.0
            elif kw in sig_lower:
                score += 1.0

            parts = re.findall(r"[a-z]+|[A-Z][a-z]*", sym.name)
            for part in parts:
                if kw == part.lower():
                    score += 1.5

        return score

    def _flatten_symbols(self, symbols: list[Symbol]) -> list[Symbol]:
        result: list[Symbol] = []
        for sym in symbols:
            result.append(sym)
            if sym.children:
                result.extend(self._flatten_symbols(sym.children))
        return result

    def _extract_block(
        self, file_path: str, sym: Symbol, context_lines: int,
    ) -> CodeBlock | None:
        try:
            lines = Path(file_path).read_text(errors="replace").split("\n")
        except OSError:
            return None

        start = max(0, sym.line_start - 1 - context_lines)
        end = min(len(lines), sym.line_end + context_lines)
        source = "\n".join(lines[start:end])

        return CodeBlock(
            file_path=file_path,
            symbol_name=sym.name,
            line_start=start + 1,
            line_end=end,
            source=source,
        )
