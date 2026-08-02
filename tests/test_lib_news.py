import pytest

from lib.news import summarize
from lib.sentiment import analyze_sentiment


def test_summarize_news(monkeypatch: pytest.MonkeyPatch) -> None:
    data = {"is_news": True, "summary": "A summary.", "reason": "News content."}
    monkeypatch.setattr("lib.news.chat", lambda *_: data)
    result = summarize("Some text")
    assert result["is_news"] is True
    assert result["summary"] == "A summary."
    assert result["reason"] == "News content."


def test_summarize_non_news(monkeypatch: pytest.MonkeyPatch) -> None:
    data = {"is_news": False, "summary": None, "reason": "Not news content."}
    monkeypatch.setattr("lib.news.chat", lambda *_: data)
    result = summarize("Some text")
    assert result["is_news"] is False
    assert result["summary"] is None


def test_analyze_sentiment_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    data = {"sentiment": "positive", "confidence": 0.9, "reason": "Good news."}
    monkeypatch.setattr("lib.sentiment.chat", lambda *_: data)
    result = analyze_sentiment("Some summary")
    assert result["sentiment"] == "positive"
    assert result["confidence"] == 0.9


def test_analyze_sentiment_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lib.sentiment.chat", lambda *_: {})
    result = analyze_sentiment("Some summary")
    assert result["sentiment"] == "neutral"
    assert result["confidence"] == 0.0
