from fastapi.testclient import TestClient
import httpx
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
            "input_price_per_token": 0.0000005,
            "output_price_per_token": 0.0000015,
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
    assert response.json()["input_price_per_token"] == 0.0000005
    assert response.json()["output_price_per_token"] == 0.0000015
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
            "input_price_per_token": 0.0000005,
            "output_price_per_token": 0.0000015,
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
    assert response.json()["input_price_per_token"] == 0.0000005
    assert response.json()["output_price_per_token"] == 0.0000015
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


_COST_REPORT: dict[str, object] = {
    "model": "openai/gpt-4o",
    "platform": "openai",
    "found": True,
    "prompt_tokens": 1000,
    "completion_tokens": 2000,
    "total_tokens": 3000,
    "input_price_per_token": 0.0000025,
    "output_price_per_token": 0.00001,
    "input_price_per_m": 2.5,
    "output_price_per_m": 10.0,
    "prompt_tokens_cost": 0.0025,
    "completion_tokens_cost": 0.02,
    "cost": 0.0225,
    "currency": "USD",
    "source": "openrouter",
}

_COST_PRICING: dict[str, object] = {
    "model": "openai/gpt-4o",
    "model_name": "OpenAI: GPT-4o",
    "provider": "openai",
    "input_price_per_token": 0.0000025,
    "output_price_per_token": 0.00001,
    "currency": "USD",
    "context_window": 128000,
    "source": "openrouter",
}


def test_cost_get(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "api.routes.cost.cost_report",
        lambda model, platform, usage: _COST_REPORT,
    )
    response = client.get(
        "/cost",
        params={"model": "gpt-4o", "prompt_tokens": 1000, "completion_tokens": 2000},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "openai/gpt-4o"
    assert body["found"] is True
    assert body["prompt_tokens_cost"] == 0.0025
    assert body["completion_tokens_cost"] == 0.02
    assert body["cost"] == 0.0225
    assert body["input_price_per_m"] == 2.5
    assert body["output_price_per_m"] == 10.0
    assert body["source"] == "openrouter"


def test_cost_post(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "api.routes.cost.cost_report",
        lambda model, platform, usage: _COST_REPORT,
    )
    response = client.post(
        "/cost",
        json={
            "model": "gpt-4o",
            "prompt_tokens": 1000,
            "completion_tokens": 2000,
            "platform": "openai",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["platform"] == "openai"
    assert body["total_tokens"] == 3000
    assert body["cost"] == 0.0225


def test_cost_get_platform_falls_back_to_bare_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def lookup(model: str, platform: str | None = None) -> list[dict[str, object]]:
        return [] if platform is not None else [_COST_PRICING]

    monkeypatch.setattr("lib.cost.find_model_pricing", lookup)
    response = client.get(
        "/cost",
        params={
            "model": "gpt-4o",
            "prompt_tokens": 1000,
            "completion_tokens": 2000,
            "platform": "groq",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert body["cost"] == 0.0225
    assert body["source"] == "openrouter"


def test_cost_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "api.routes.cost.cost_report",
        lambda model, platform, usage: {
            "model": "unknown-model",
            "platform": "groq",
            "found": False,
            "prompt_tokens": 1000,
            "completion_tokens": 2000,
            "total_tokens": 3000,
            "input_price_per_token": None,
            "output_price_per_token": None,
            "input_price_per_m": None,
            "output_price_per_m": None,
            "prompt_tokens_cost": 0.0,
            "completion_tokens_cost": 0.0,
            "cost": 0.0,
            "currency": "USD",
            "source": "",
        },
    )
    response = client.get(
        "/cost",
        params={"model": "unknown-model", "prompt_tokens": 1000, "completion_tokens": 2000},
    )
    assert response.status_code == 404
    assert response.json()["detail"]["message"] == "Pricing not found"


def test_cost_upstream_down(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(model: str, platform: str | None, usage: object) -> None:
        raise httpx.ConnectError(
            "network error", request=httpx.Request("GET", "https://example.com/pricing")
        )

    monkeypatch.setattr("api.routes.cost.cost_report", boom)
    response = client.get(
        "/cost",
        params={"model": "gpt-4o", "prompt_tokens": 1000, "completion_tokens": 2000},
    )
    assert response.status_code == 503
    assert "Unable to fetch pricing data" in response.json()["detail"]


def test_cost_post_negative_tokens() -> None:
    response = client.post(
        "/cost",
        json={"model": "gpt-4o", "prompt_tokens": -1, "completion_tokens": 2000},
    )
    assert response.status_code == 422


def test_cost_get_negative_tokens() -> None:
    response = client.get(
        "/cost",
        params={"model": "gpt-4o", "prompt_tokens": 1000, "completion_tokens": -1},
    )
    assert response.status_code == 422


def test_cost_get_missing_model() -> None:
    response = client.get("/cost", params={"prompt_tokens": 1000, "completion_tokens": 2000})
    assert response.status_code == 422
