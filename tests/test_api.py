from fastapi.testclient import TestClient
import pytest

from api.main import app
from lib.config import MissingApiKeyError
from lib.llm import LLMError

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_summarize_news(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lib.news.summarize",
        lambda text: {"is_news": True, "summary": "A summary.", "reason": "News."},
    )
    response = client.post("/summarize", json={"text": "Some news text"})
    assert response.status_code == 200
    assert response.json()["is_news"] is True
    assert response.json()["summary"] == "A summary."


def test_summarize_non_news(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lib.news.summarize",
        lambda text: {"is_news": False, "summary": None, "reason": "Not news."},
    )
    response = client.post("/summarize", json={"text": "A recipe"})
    assert response.status_code == 200
    assert response.json()["is_news"] is False
    assert response.json()["summary"] is None


def test_analyse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lib.sentiment.analyze_sentiment",
        lambda text: {"sentiment": "positive", "confidence": 0.9, "reason": "Good."},
    )
    response = client.post("/analyse", json={"text": "A positive summary"})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"
    assert response.json()["confidence"] == 0.9


def test_summarize_empty_text() -> None:
    response = client.post("/summarize", json={"text": ""})
    assert response.status_code == 422


def test_analyse_empty_text() -> None:
    response = client.post("/analyse", json={"text": ""})
    assert response.status_code == 422


def test_summarize_missing_text() -> None:
    response = client.post("/summarize", json={})
    assert response.status_code == 422


def test_summarize_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(text: str) -> dict[str, object]:
        raise LLMError("LLM request failed")

    monkeypatch.setattr("lib.news.summarize", fail)
    response = client.post("/summarize", json={"text": "Some text"})
    assert response.status_code == 502


def test_analyse_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(text: str) -> dict[str, object]:
        raise LLMError("LLM request failed")

    monkeypatch.setattr("lib.sentiment.analyze_sentiment", fail)
    response = client.post("/analyse", json={"text": "Some text"})
    assert response.status_code == 502


def test_summarize_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(text: str) -> dict[str, object]:
        raise MissingApiKeyError("LLM_API_KEY is not set")

    monkeypatch.setattr("lib.news.summarize", fail)
    response = client.post("/summarize", json={"text": "Some text"})
    assert response.status_code == 500


def test_analyse_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(text: str) -> dict[str, object]:
        raise MissingApiKeyError("LLM_API_KEY is not set")

    monkeypatch.setattr("lib.sentiment.analyze_sentiment", fail)
    response = client.post("/analyse", json={"text": "Some text"})
    assert response.status_code == 500
