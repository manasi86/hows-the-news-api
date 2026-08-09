from collections.abc import Mapping
from typing import cast

import pytest

from lib.llm import ChatResult, JsonObject, TokenUsage
from lib.news import summarize
from lib.sentiment import analyze_sentiment


def _chat_result(data: Mapping[str, object]) -> ChatResult:
    return ChatResult(
        data=cast(JsonObject, data),
        usage=TokenUsage(prompt_tokens=1_000_000, completion_tokens=500_000),
        model="openai/gpt-oss-120b",
        platform="groq",
    )


def test_summarize_news(monkeypatch: pytest.MonkeyPatch) -> None:
    data = {"is_news": True, "summary": "A summary.", "reason": "News content."}
    monkeypatch.setattr("lib.news.chat", lambda *_: _chat_result(data))
    # _disable_pricing(monkeypatch)
    result = summarize("Some text")
    assert result["is_news"] is True
    assert result["summary"] == "A summary."
    assert result["reason"] == "News content."
    assert result["model"] == "openai/gpt-oss-120b"
    assert result["platform"] == "groq"
    assert result["prompt_tokens"] == 1_000_000
    assert result["completion_tokens"] == 500_000
    assert result["total_tokens"] == 1_500_000
    assert result["prompt_tokens_cost"] == 0.039
    assert result["completion_tokens_cost"] == 0.050
    assert result["cost"] == 0.089
    assert result["source"] == ""


def test_summarize_non_news(monkeypatch: pytest.MonkeyPatch) -> None:
    data = {"is_news": False, "summary": None, "reason": "Not news content."}
    monkeypatch.setattr("lib.news.chat", lambda *_: _chat_result(data))
    # _disable_pricing(monkeypatch)
    result = summarize("Some text")
    assert result["is_news"] is False
    assert result["summary"] is None


def test_analyze_sentiment_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    data = {"sentiment": "positive", "confidence": 0.9, "reason": "Good news."}
    monkeypatch.setattr("lib.sentiment.chat", lambda *_: _chat_result(data))
    # _disable_pricing(monkeypatch)
    result = analyze_sentiment("Some summary")
    assert result["sentiment"] == "positive"
    assert result["confidence"] == 0.9
    assert result["model"] == "openai/gpt-oss-120b"
    assert result["platform"] == "groq"
    assert result["prompt_tokens"] == 1_000_000
    assert result["completion_tokens"] == 500_000
    assert result["total_tokens"] == 1_500_000
    assert result["prompt_tokens_cost"] == 0.039
    assert result["completion_tokens_cost"] == 0.050
    assert result["cost"] == 0.089
    assert result["source"] == ""


def test_analyze_sentiment_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lib.sentiment.chat", lambda *_: _chat_result({}))
    # _disable_pricing(monkeypatch)
    result = analyze_sentiment("Some summary")
    assert result["sentiment"] == "neutral"
    assert result["confidence"] == 0.0


def test_summarize_url(monkeypatch: pytest.MonkeyPatch) -> None:
    data = {"is_news": True, "summary": "A summary.", "reason": "News content."}
    monkeypatch.setattr("lib.news.fetch_article", lambda url: ("Article text", None))
    monkeypatch.setattr("lib.news.chat", lambda *_: _chat_result(data))
    # _disable_pricing(monkeypatch)
    result = summarize(url="https://example.com/news")
    assert result["is_news"] is True
    assert result["summary"] == "A summary."


def test_summarize_url_fetch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lib.news.fetch_article", lambda url: (None, "could not fetch"))
    result = summarize(url="https://example.com/news")
    assert result["is_news"] is False
    assert result["summary"] is None
    assert result["reason"] == "Could not extract the news article from the URL: could not fetch"
    assert result["prompt_tokens"] == 0
    assert result["completion_tokens"] == 0
    assert result["total_tokens"] == 0
    assert result["prompt_tokens_cost"] == 0.0
    assert result["completion_tokens_cost"] == 0.0
    assert result["cost"] == 0.0


def test_summarize_url_empty_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lib.news.fetch_article", lambda url: (None, None))
    result = summarize(url="https://example.com/news")
    assert result["is_news"] is False
    assert result["summary"] is None
    assert result["reason"] == "Could not extract the news article from the URL"
    assert result["prompt_tokens"] == 0
    assert result["completion_tokens"] == 0
    assert result["total_tokens"] == 0
    assert result["prompt_tokens_cost"] == 0.0
    assert result["completion_tokens_cost"] == 0.0
    assert result["cost"] == 0.0


def test_summarize_no_input() -> None:
    with pytest.raises(ValueError, match="Provide exactly one of 'text' or 'url'"):
        summarize()
