from fastapi.testclient import TestClient
import pytest

from api.main import app
from lib.config import MissingApiKeyError
from lib.llm import LLMError

# from lib.pricing import PricingError

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_summarize_news(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lib.news.summarize",
        lambda text, url: {
            "is_news": True,
            "summary": "A summary.",
            "reason": "News.",
            "model": "openai/gpt-oss-120b",
            "platform": "groq",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_cost": 0.000004,
            "completion_tokens_cost": 0.000005,
            "cost": 0.000009,
            "input_price_per_m": 0.5,
            "output_price_per_m": 1.5,
            "source": "https://llm-stats.com/models/gpt-oss-120b#pricing",
        },
    )
    response = client.post("/summarize", json={"text": "Some news text"})
    assert response.status_code == 200
    assert response.json()["is_news"] is True
    assert response.json()["summary"] == "A summary."
    assert response.json()["model"] == "openai/gpt-oss-120b"
    assert response.json()["platform"] == "groq"
    assert response.json()["prompt_tokens"] == 100
    assert response.json()["completion_tokens"] == 50
    assert response.json()["total_tokens"] == 150
    assert response.json()["prompt_tokens_cost"] == 0.000004
    assert response.json()["completion_tokens_cost"] == 0.000005
    assert response.json()["cost"] == 0.000009
    assert response.json()["input_price_per_m"] == 0.5
    assert response.json()["output_price_per_m"] == 1.5
    assert response.json()["source"] == "https://llm-stats.com/models/gpt-oss-120b#pricing"


def test_summarize_non_news(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lib.news.summarize",
        lambda text, url: {"is_news": False, "summary": None, "reason": "Not news."},
    )
    response = client.post("/summarize", json={"text": "A recipe"})
    assert response.status_code == 200
    assert response.json()["is_news"] is False
    assert response.json()["summary"] is None
    assert response.json()["model"] == ""
    assert response.json()["platform"] is None
    assert response.json()["prompt_tokens"] == 0
    assert response.json()["completion_tokens"] == 0
    assert response.json()["total_tokens"] == 0
    assert response.json()["prompt_tokens_cost"] == 0.0
    assert response.json()["completion_tokens_cost"] == 0.0
    assert response.json()["cost"] == 0.0
    assert response.json()["source"] == ""


def test_analyse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lib.sentiment.analyze_sentiment",
        lambda text: {
            "sentiment": "positive",
            "confidence": 0.9,
            "reason": "Good.",
            "model": "openai/gpt-oss-120b",
            "platform": "groq",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_cost": 0.000004,
            "completion_tokens_cost": 0.000005,
            "cost": 0.000009,
            "input_price_per_m": 0.5,
            "output_price_per_m": 1.5,
            "source": "https://example.com/pricing",
        },
    )
    response = client.post("/analyse", json={"text": "A positive summary"})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"
    assert response.json()["confidence"] == 0.9
    assert response.json()["model"] == "openai/gpt-oss-120b"
    assert response.json()["platform"] == "groq"
    assert response.json()["prompt_tokens"] == 100
    assert response.json()["completion_tokens"] == 50
    assert response.json()["total_tokens"] == 150
    assert response.json()["prompt_tokens_cost"] == 0.000004
    assert response.json()["completion_tokens_cost"] == 0.000005
    assert response.json()["cost"] == 0.000009
    assert response.json()["input_price_per_m"] == 0.5
    assert response.json()["output_price_per_m"] == 1.5
    assert response.json()["source"] == "https://example.com/pricing"


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
    def fail(text: str, url: str | None) -> dict[str, object]:
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
    def fail(text: str, url: str | None) -> dict[str, object]:
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
