from fastapi.testclient import TestClient
import httpx
import pytest

from api.main import app
from lib.pricing import find_model_pricing, load_pricing_data, normalise

MOCK_DATA: dict[str, list[dict[str, object]]] = {
    "data": [
        {
            "id": "openai/gpt-4o",
            "name": "OpenAI: GPT-4o",
            "context_length": 128000,
            "pricing": {"prompt": "0.0000025", "completion": "0.00001"},
        },
        {
            "id": "openai/gpt-3.5-turbo",
            "name": "OpenAI: GPT-3.5 Turbo",
            "context_length": 4096,
            "pricing": {"prompt": "0.0000005", "completion": "0.0000015"},
        },
        {
            "id": "anthropic/claude-3-5-sonnet",
            "name": "Anthropic: Claude 3.5 Sonnet",
            "context_length": 200000,
            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
        },
        {
            "id": "azure-openai/gpt-4o",
            "name": "Azure: GPT-4o",
            "context_length": 128000,
            "pricing": {"prompt": "0.0000026", "completion": "0.0000105"},
        },
        {
            "id": "~deepseek/deepseek-v4-flash",
            "name": "DeepSeek: V4 Flash",
            "context_length": 131072,
            "pricing": {"prompt": "0.0000002", "completion": "0.0000008"},
        },
        {
            "id": "google/gemini-2.5-flash:free",
            "name": "Google: Gemini 2.5 Flash (free)",
            "context_length": 1048576,
            "pricing": {"prompt": "0", "completion": "0"},
        },
    ]
}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr("lib.pricing.load_pricing_data", lambda: MOCK_DATA)
    return TestClient(app)


@pytest.fixture
def upstream_down(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    def boom() -> None:
        raise httpx.ConnectError(
            "network error", request=httpx.Request("GET", "https://example.com/pricing")
        )

    monkeypatch.setattr("lib.pricing.load_pricing_data", boom)
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_pricing_bare_name_matches_across_providers(client: TestClient) -> None:
    response = client.get("/pricing", params={"model": "gpt-4o"})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "gpt-4o"
    assert len(body["providers"]) == 2
    provider_ids = {p["provider"] for p in body["providers"]}
    assert provider_ids == {"openai", "azure-openai"}
    assert all(p["input_price_per_token"] is not None for p in body["providers"])
    assert all(p["source"] == "openrouter" for p in body["providers"])


def test_pricing_full_id_match(client: TestClient) -> None:
    response = client.get("/pricing", params={"model": "openai/gpt-4o"})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "openai/gpt-4o"
    assert body["providers"][0]["provider"] == "openai"
    assert body["providers"][0]["input_price_per_token"] == 0.0000025


def test_pricing_provider_filter_returns_single(client: TestClient) -> None:
    response = client.get(
        "/pricing",
        params={"model": "gpt-4o", "provider": "openai"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "openai"
    assert body["input_price_per_token"] == 0.0000025
    assert body["output_price_per_token"] == 0.00001


def test_pricing_provider_exact_match_preferred(client: TestClient) -> None:
    response = client.get(
        "/pricing",
        params={"model": "gpt-4o", "provider": "openai"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "openai"
    assert "azure" not in body["provider"]


def test_pricing_provider_fuzzy_match(client: TestClient) -> None:
    response = client.get(
        "/pricing",
        params={"model": "gpt-4o", "provider": "azure"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "azure-openai"


def test_pricing_provider_matches_id_prefix(client: TestClient) -> None:
    response = client.get(
        "/pricing",
        params={"model": "claude-3-5-sonnet", "provider": "anthropic"},
    )
    assert response.status_code == 200
    assert response.json()["provider"] == "anthropic"


def test_pricing_name_match(client: TestClient) -> None:
    response = client.get(
        "/pricing",
        params={"model": "Anthropic: Claude 3.5 Sonnet", "provider": "anthropic"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "anthropic/claude-3-5-sonnet"
    assert body["model_name"] == "Anthropic: Claude 3.5 Sonnet"


def test_pricing_model_name_case_insensitive(client: TestClient) -> None:
    response = client.get(
        "/pricing",
        params={"model": "GPT-4O", "provider": "OpenAI"},
    )
    assert response.status_code == 200
    assert response.json()["provider"] == "openai"


def test_pricing_model_not_found(client: TestClient) -> None:
    response = client.get("/pricing", params={"model": "does-not-exist"})
    assert response.status_code == 404
    body = response.json()
    assert body["detail"]["message"] == "Pricing not found"
    assert body["detail"]["model"] == "does-not-exist"


def test_pricing_provider_not_found(client: TestClient) -> None:
    response = client.get(
        "/pricing",
        params={"model": "gpt-4o", "provider": "google"},
    )
    assert response.status_code == 404
    assert response.json()["detail"]["message"] == "Pricing not found"


def test_pricing_upstream_down_returns_503(upstream_down: TestClient) -> None:
    response = upstream_down.get("/pricing", params={"model": "gpt-4o"})
    assert response.status_code == 503
    assert "Unable to fetch pricing data" in response.json()["detail"]


def test_normalise() -> None:
    assert normalise("GPT-4o") == "gpt-4o"
    assert normalise("  Claude 3.5 Sonnet  ") == "claude 3.5 sonnet"
    assert normalise("my_model") == "my-model"


class FakePricingResponse:
    """Minimal stand-in for an ``httpx.Response`` used in pricing tests."""

    def __init__(self, json_data: object) -> None:
        self._json_data = json_data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> object:
        return self._json_data


def test_load_pricing_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lib.pricing.httpx.get",
        lambda *args, **kwargs: FakePricingResponse({"data": [{"id": "openai/gpt-4o"}]}),
    )
    assert load_pricing_data() == {"data": [{"id": "openai/gpt-4o"}]}


def test_load_pricing_data_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise httpx.ConnectError(
            "network error", request=httpx.Request("GET", "https://example.com/pricing")
        )

    monkeypatch.setattr("lib.pricing.httpx.get", boom)
    with pytest.raises(httpx.HTTPError):
        load_pricing_data()


def test_find_model_pricing_direct_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lib.pricing.load_pricing_data", lambda: MOCK_DATA)
    results = find_model_pricing("gpt-3.5-turbo")
    assert len(results) == 1
    assert results[0]["provider"] == "openai"
    assert results[0]["input_price_per_token"] == 0.0000005


def test_find_model_pricing_full_id_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lib.pricing.load_pricing_data", lambda: MOCK_DATA)
    results = find_model_pricing("openai/gpt-4o")
    assert len(results) == 1
    assert results[0]["model"] == "openai/gpt-4o"
    assert results[0]["input_price_per_token"] == 0.0000025


def test_find_model_pricing_variant_suffix_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lib.pricing.load_pricing_data", lambda: MOCK_DATA)
    results = find_model_pricing("gemini-2.5-flash")
    assert len(results) == 1
    assert results[0]["model"] == "google/gemini-2.5-flash:free"
    assert results[0]["input_price_per_token"] == 0.0
    assert results[0]["output_price_per_token"] == 0.0


def test_find_model_pricing_tilde_prefix_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lib.pricing.load_pricing_data", lambda: MOCK_DATA)
    results = find_model_pricing("deepseek-v4-flash", "deepseek")
    assert len(results) == 1
    assert results[0]["provider"] == "deepseek"
    assert results[0]["model"] == "deepseek/deepseek-v4-flash"


def test_find_model_pricing_name_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lib.pricing.load_pricing_data", lambda: MOCK_DATA)
    results = find_model_pricing("Anthropic: Claude 3.5 Sonnet")
    assert len(results) == 1
    assert results[0]["model"] == "anthropic/claude-3-5-sonnet"


def test_find_model_pricing_missing_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    data: dict[str, list[dict[str, object]]] = {
        "data": [
            {
                "id": "example/foo",
                "name": "Example Foo",
                "context_length": 1000,
                "pricing": {},
            },
            {
                "id": "example/bar",
                "name": "Example Bar",
                "context_length": 1000,
                "pricing": {"prompt": "not-a-number", "completion": ""},
            },
        ]
    }
    monkeypatch.setattr("lib.pricing.load_pricing_data", lambda: data)
    results = find_model_pricing("foo")
    assert len(results) == 1
    assert results[0]["input_price_per_token"] is None
    assert results[0]["output_price_per_token"] is None
    results = find_model_pricing("bar")
    assert results[0]["input_price_per_token"] is None
    assert results[0]["output_price_per_token"] is None


def test_find_model_pricing_missing_data_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lib.pricing.load_pricing_data", lambda: {"total_count": 0})
    assert find_model_pricing("unknown-model") == []


def test_find_model_pricing_no_match_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lib.pricing.load_pricing_data", lambda: MOCK_DATA)
    assert find_model_pricing("unknown-model") == []
