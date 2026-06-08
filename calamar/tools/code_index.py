"""Code index tools — repo map and code locate for the agent."""

from __future__ import annotations

from calamar.tools.base import ToolParameter, ToolResult, ToolSpec


class CodeMapTool:
    """Generate a structural map of the codebase using tree-sitter."""

    def __init__(self, working_dir: str = ".") -> None:
        self._working_dir = working_dir
        self._repo_map = None

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="code_map",
            description=(
                "Generate a structural map of the codebase showing all classes, "
                "functions, and methods with their signatures. Use this to understand "
                "the overall code structure before making changes. Supports Python, "
                "JavaScript, TypeScript, Java, Go, and Rust."
            ),
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description=(
                        "Subdirectory to map (relative to project root). "
                        "Use '.' for the entire project."
                    ),
                    required=False,
                    default=".",
                ),
                ToolParameter(
                    name="max_tokens",
                    type="integer",
                    description="Maximum token budget for the map (default: 2000).",
                    required=False,
                    default=2000,
                ),
            ],
        )

    async def execute(
        self,
        path: str = ".",
        max_tokens: int = 2000,
        **_: object,
    ) -> ToolResult:
        try:
            from calamar.code_index.repo_map import RepoMap
        except ImportError:
            return ToolResult(
                error="tree-sitter not installed. Run: pip install calamar[code-index]",
            )

        try:
            if self._repo_map is None:
                self._repo_map = RepoMap(self._working_dir)

            if path and path != ".":
                from pathlib import Path
                scan_root = Path(self._working_dir) / path
                if not scan_root.exists():
                    return ToolResult(error=f"Path not found: {path}")
                sub_map = RepoMap(str(scan_root))
                text = sub_map.render(max_tokens=max_tokens)
            else:
                text = self._repo_map.render(max_tokens=max_tokens)

            if not text.strip():
                return ToolResult(
                    output="No supported source files found.",
                    metadata={"files": 0},
                )

            file_count = text.count("\n") - text.count("\n  ")
            return ToolResult(
                output=text,
                metadata={"files": file_count},
            )
        except Exception as e:
            return ToolResult(error=f"Failed to generate code map: {e}")


class CodeLocateTool:
    """Locate relevant code blocks by query or symbol name."""

    def __init__(self, working_dir: str = ".") -> None:
        self._working_dir = working_dir
        self._locator = None

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="code_locate",
            description=(
                "Find relevant code blocks by natural language query or symbol name. "
                "Returns the actual source code of matching functions, classes, or "
                "methods with surrounding context. Use this when you need to read "
                "specific code based on what it does, not where it is."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description=(
                        "What to search for — a symbol name (e.g. 'AgentLoop'), "
                        "a concept (e.g. 'middleware pipeline'), or a task "
                        "(e.g. 'authentication handling')."
                    ),
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum number of code blocks to return (default: 5).",
                    required=False,
                    default=5,
                ),
            ],
        )

    async def execute(
        self,
        query: str = "",
        max_results: int = 5,
        **_: object,
    ) -> ToolResult:
        if not query:
            return ToolResult(error="query is required.")

        try:
            from calamar.code_index.locator import CodeLocator
            from calamar.code_index.repo_map import RepoMap
        except ImportError:
            return ToolResult(
                error="tree-sitter not installed. Run: pip install calamar[code-index]",
            )

        try:
            if self._locator is None:
                rm = RepoMap(self._working_dir)
                self._locator = CodeLocator(rm)

            result = self._locator.locate(query, max_results=max_results)

            if not result.blocks:
                return ToolResult(output=f"No code found matching: {query}")

            parts: list[str] = []
            for block in result.blocks:
                header = (
                    f"--- {block.file_path}:{block.line_start}-{block.line_end} "
                    f"({block.symbol_name}) ---"
                )
                parts.append(header)
                parts.append(block.source)
                parts.append("")

            if result.repo_map_excerpt:
                parts.append("--- Related structure ---")
                parts.append(result.repo_map_excerpt)

            return ToolResult(
                output="\n".join(parts),
                metadata={"matches": len(result.blocks)},
            )
        except Exception as e:
            return ToolResult(error=f"Code locate failed: {e}")
