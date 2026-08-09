import httpx
import pytest

from lib.cost import cost_report, no_pricing_report
from lib.llm import TokenUsage


def _pricing() -> dict[str, object]:
    return {
        "model": "openai/gpt-4o",
        "model_name": "OpenAI: GPT-4o",
        "provider": "openai",
        "input_price_per_token": 0.0000025,
        "output_price_per_token": 0.00001,
        "currency": "USD",
        "context_window": 128000,
        "source": "openrouter",
    }


def test_cost_report_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lib.cost.find_model_pricing", lambda model, platform: [_pricing()])
    usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    report = cost_report("openai/gpt-4o", "openai", usage)
    assert report["model"] == "openai/gpt-4o"
    assert report["platform"] == "openai"
    assert report["found"] is True
    assert report["prompt_tokens"] == 1_000_000
    assert report["completion_tokens"] == 1_000_000
    assert report["total_tokens"] == 2_000_000
    assert report["input_price_per_token"] == 0.0000025
    assert report["output_price_per_token"] == 0.00001
    assert report["input_price_per_m"] == 2.5
    assert report["output_price_per_m"] == 10.0
    assert report["prompt_tokens_cost"] == 2.5
    assert report["completion_tokens_cost"] == 10.0
    assert report["cost"] == 12.5
    assert report["currency"] == "USD"
    assert report["source"] == "openrouter"


def test_cost_report_rounds_to_six_decimals(monkeypatch: pytest.MonkeyPatch) -> None:
    pricing = _pricing()
    pricing["input_price_per_token"] = 0.0000001
    pricing["output_price_per_token"] = 0.0000002
    monkeypatch.setattr("lib.cost.find_model_pricing", lambda model, platform: [pricing])
    usage = TokenUsage(prompt_tokens=999_999, completion_tokens=777_777)
    report = cost_report("openai/gpt-4o", None, usage)
    assert report["prompt_tokens_cost"] == 0.1
    assert report["completion_tokens_cost"] == 0.155555
    assert report["cost"] == 0.255555


def test_cost_report_no_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lib.cost.find_model_pricing", lambda model, platform=None: [])
    usage = TokenUsage(prompt_tokens=10, completion_tokens=20)
    report = cost_report("unknown-model", "groq", usage)
    assert report["model"] == "unknown-model"
    assert report["platform"] == "groq"
    assert report["found"] is False
    assert report["input_price_per_token"] is None
    assert report["output_price_per_token"] is None
    assert report["prompt_tokens_cost"] == 0.0
    assert report["completion_tokens_cost"] == 0.0
    assert report["cost"] == 0.0
    assert report["source"] == ""


def test_cost_report_falls_back_to_bare_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def lookup(model: str, platform: str | None = None) -> list[dict[str, object]]:
        return [] if platform is not None else [_pricing()]

    monkeypatch.setattr("lib.cost.find_model_pricing", lookup)
    usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    report = cost_report("openai/gpt-oss-120b", "groq", usage)
    assert report["found"] is True
    assert report["model"] == "openai/gpt-4o"
    assert report["platform"] == "groq"
    assert report["input_price_per_token"] == 0.0000025
    assert report["cost"] == 12.5
    assert report["source"] == "openrouter"


def test_cost_report_missing_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    pricing = _pricing()
    pricing["input_price_per_token"] = None
    pricing["output_price_per_token"] = None
    monkeypatch.setattr("lib.cost.find_model_pricing", lambda model, platform: [pricing])
    usage = TokenUsage(prompt_tokens=10, completion_tokens=20)
    report = cost_report("openai/gpt-4o", None, usage)
    assert report["found"] is False
    assert report["input_price_per_token"] is None
    assert report["output_price_per_token"] is None
    assert report["cost"] == 0.0
    assert report["source"] == ""


def test_cost_report_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(model: str, platform: str | None) -> list[object]:
        raise httpx.ConnectError(
            "network error", request=httpx.Request("GET", "https://example.com/pricing")
        )

    monkeypatch.setattr("lib.cost.find_model_pricing", boom)
    usage = TokenUsage(prompt_tokens=10, completion_tokens=20)
    with pytest.raises(httpx.HTTPError):
        cost_report("openai/gpt-4o", None, usage)


def test_no_pricing_report() -> None:
    usage = TokenUsage(prompt_tokens=10, completion_tokens=20)
    report = no_pricing_report("openai/gpt-4o", "groq", usage)
    assert report["model"] == "openai/gpt-4o"
    assert report["platform"] == "groq"
    assert report["found"] is False
    assert report["prompt_tokens"] == 10
    assert report["completion_tokens"] == 20
    assert report["total_tokens"] == 30
    assert report["input_price_per_token"] is None
    assert report["output_price_per_token"] is None
    assert report["input_price_per_m"] is None
    assert report["output_price_per_m"] is None
    assert report["prompt_tokens_cost"] == 0.0
    assert report["completion_tokens_cost"] == 0.0
    assert report["cost"] == 0.0
    assert report["currency"] == "USD"
    assert report["source"] == ""
