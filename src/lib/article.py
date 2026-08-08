"""Fetch and extract article text from a news article URL."""

from html.parser import HTMLParser
import ipaddress
from typing import TypeAlias
from urllib.parse import urlparse

import httpx as httpx
from typing_extensions import override

from lib.config import get_settings

ExtractionResult: TypeAlias = tuple[str | None, str | None]

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; hows-the-news/1.0; +https://github.com/nalits/hows-the-news)",
}

_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tr",
    "ul",
}

_SKIP_TAGS = {"noscript", "script", "style", "svg", "template"}


class _TextExtractor(HTMLParser):
    """Extract readable text from HTML, skipping non-visible tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    @override
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    @override
    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    @override
    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._chunks.append(data)

    def text(self) -> str:
        """Return the extracted text with whitespace normalised."""
        return _normalise("".join(self._chunks))


def _normalise(text: str) -> str:
    """Collapse whitespace and drop empty lines from ``text``."""
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _extract_text(html: str) -> str:
    """Strip HTML markup and return the visible text content."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except (AssertionError, ValueError):
        return ""
    return parser.text()


def _is_blocked_host(hostname: str) -> bool:
    """Return True if ``hostname`` is a loopback or private address.

    This is a basic SSRF guard that blocks IP literals pointing at local or
    private networks, as well as the ``localhost`` hostname.
    """
    host = hostname.lower().rstrip(".")
    if host == "localhost":
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified,
    )


def fetch_article(url: str) -> ExtractionResult:
    """Fetch ``url`` and extract the article text from its HTML.

    Args:
        url: The URL of the news article to fetch.

    Returns:
        A tuple of the extracted article text and an error reason. Exactly one
        of the two values is ``None``.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        return None, "the URL must use the http or https scheme"
    if _is_blocked_host(parsed.hostname):
        return None, "the URL points to a local or private address"
    settings = get_settings()
    try:
        response = httpx.get(
            url,
            headers=_HEADERS,
            follow_redirects=True,
            timeout=settings.llm_timeout,
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "html" not in content_type.lower():
            return None, "the URL does not point to an HTML news article"
        text = _extract_text(response.text)
    except httpx.HTTPStatusError as exc:
        return None, f"the URL returned HTTP status {exc.response.status_code}"
    except httpx.RequestError:
        return None, "the URL could not be fetched"
    if not text:
        return None, "no article content could be extracted from the URL"
    return text, None
