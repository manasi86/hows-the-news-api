import httpx
import pytest

from lib import article
from lib.article import fetch_article


def _patch_get(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status_code: int = 200,
    text: str = "",
    content_type: str = "text/html",
) -> None:
    class _FakeResponse:
        @property
        def status_code(self) -> int:
            return status_code

        @property
        def headers(self) -> dict[str, str]:
            return {"content-type": content_type}

        @property
        def text(self) -> str:
            return text

        def raise_for_status(self) -> None:
            if status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"error {status_code}",
                    request=httpx.Request("GET", "https://example.com"),
                    response=httpx.Response(status_code, request=httpx.Request("GET", "https://example.com")),
                )

    monkeypatch.setattr(article.httpx, "get", lambda *args, **kwargs: _FakeResponse())


def test_fetch_article_success(monkeypatch: pytest.MonkeyPatch) -> None:
    html = (
        "<html><body><h1>Headline</h1>"
        "<script>var hidden = 1;</script>"
        "<p> First   paragraph. </p><p>Second paragraph.</p></body></html>"
    )
    _patch_get(monkeypatch, text=html)
    text, error = fetch_article("https://example.com/news")
    assert error is None
    assert text == "Headline\nFirst paragraph.\nSecond paragraph."


def test_fetch_article_rejects_non_http_scheme() -> None:
    text, error = fetch_article("ftp://example.com/news")
    assert text is None
    assert error == "the URL must use the http or https scheme"


def test_fetch_article_rejects_missing_hostname() -> None:
    text, error = fetch_article("http://")
    assert text is None
    assert error == "the URL must use the http or https scheme"


def test_fetch_article_blocks_localhost() -> None:
    text, error = fetch_article("http://localhost/news")
    assert text is None
    assert error == "the URL points to a local or private address"


def test_fetch_article_blocks_private_ip() -> None:
    text, error = fetch_article("http://10.0.0.1/news")
    assert text is None
    assert error == "the URL points to a local or private address"


def test_fetch_article_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, status_code=404)
    text, error = fetch_article("https://example.com/news")
    assert text is None
    assert error == "the URL returned HTTP status 404"


def test_fetch_article_request_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_error(*args: object, **kwargs: object) -> None:
        raise httpx.RequestError("boom")

    monkeypatch.setattr(article.httpx, "get", raise_error)
    text, error = fetch_article("https://example.com/news")
    assert text is None
    assert error == "the URL could not be fetched"


def test_fetch_article_rejects_non_html_content(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, content_type="application/pdf")
    text, error = fetch_article("https://example.com/news.pdf")
    assert text is None
    assert error == "the URL does not point to an HTML news article"


def test_fetch_article_no_extracted_content(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, text="<html><body><script>only script</script></body></html>")
    text, error = fetch_article("https://example.com/news")
    assert text is None
    assert error == "no article content could be extracted from the URL"


def test_extract_text_handles_parser_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _RaisingExtractor:
        def __init__(self) -> None:
            pass

        def feed(self, html: str) -> None:
            raise AssertionError("boom")

        def text(self) -> str:
            return ""

    monkeypatch.setattr(article, "_TextExtractor", _RaisingExtractor)
    assert article._extract_text("<html>") == ""
