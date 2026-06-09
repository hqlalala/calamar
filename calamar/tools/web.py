"""Web fetch tool — retrieve and extract text from URLs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

import httpx

from calamar.tools.base import ToolParameter, ToolResult, ToolSpec


class _HTMLTextExtractor(HTMLParser):
    """Extract visible text from HTML, stripping tags and scripts."""

    _SKIP_TAGS = {"script", "style", "noscript", "head", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._parts.append(text)

    def get_text(self) -> str:
        raw = " ".join(self._parts)
        return re.sub(r"\s+", " ", raw).strip()


@dataclass
class WebFetchTool:
    """Fetch a URL and return its text content."""

    _max_length: int = 8000
    _timeout: float = 15.0

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="web_fetch",
            description=(
                "Fetch a URL and return its text content. "
                "Useful for reading documentation, API references, "
                "error messages, and other web content."
            ),
            parameters=[
                ToolParameter(
                    name="url",
                    type="string",
                    description="The URL to fetch",
                ),
                ToolParameter(
                    name="extract_text",
                    type="boolean",
                    description="Strip HTML tags and return plain text (default: true)",
                    required=False,
                    default=True,
                ),
            ],
        )

    async def execute(self, url: str = "", extract_text: bool = True) -> ToolResult:
        if not url:
            return ToolResult(error="URL is required")

        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        try:
            async with httpx.AsyncClient(
                follow_redirects=True, timeout=self._timeout,
            ) as client:
                response = await client.get(
                    url,
                    headers={"User-Agent": "Calamar/0.2 (AI Agent)"},
                )
                response.raise_for_status()
        except httpx.TimeoutException:
            return ToolResult(error=f"Request timed out after {self._timeout}s")
        except httpx.HTTPStatusError as exc:
            return ToolResult(error=f"HTTP {exc.response.status_code}: {url}")
        except httpx.RequestError as exc:
            return ToolResult(error=f"Request failed: {exc}")

        content_type = response.headers.get("content-type", "")
        body = response.text

        if extract_text and "html" in content_type:
            extractor = _HTMLTextExtractor()
            try:
                extractor.feed(body)
                body = extractor.get_text()
            except Exception:
                pass

        if len(body) > self._max_length:
            body = body[:self._max_length] + "\n...[truncated]..."

        return ToolResult(
            output=body,
            metadata={"url": str(response.url), "status_code": response.status_code},
        )
