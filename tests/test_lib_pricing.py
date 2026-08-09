from fastapi.testclient import TestClient
import httpx
import pytest

from api.main import app
from lib.pricing import find_model_pricing, load_pricing_data, normalise

MOCK_DATA: list[dict[str, object]] = [
    {
        "id": "openai",
        "name": "OpenAI",
        "models": [
            {
                "id": "gpt-4o",
                "name": "GPT-4o",
                "match": {"or": [{"starts_with": "gpt-4o"}]},
                "prices": {"input_mtok": 2.5, "output_mtok": 10.0},
                "context_window": 128000,
                "prices_checked": "2026-01-01",
            },
            {
                "id": "gpt-3.5-turbo",
                "name": "GPT-3.5 Turbo",
                "match": {"or": [{"equals": "gpt-3.5-turbo"}]},
                "prices": {"input_mtok": 0.5, "output_mtok": 1.5},
                "context_window": 4096,
                "prices_checked": "2026-01-01",
            },
        ],
    },
    {
        "id": "anthropic",
        "name": "Anthropic",
        "models": [
            {
                "id": "claude-3-5-sonnet",
                "name": "Claude 3.5 Sonnet",
                "match": {
                    "or": [
                        {"starts_with": "claude-3-5-sonnet"},
                        {"equals": "claude-3.5-sonnet"},
                    ]
                },
                "prices": {"input_mtok": 3.0, "output_mtok": 15.0},
                "context_window": 200000,
                "prices_checked": "2026-01-01",
            }
        ],
    },
    {
        "id": "azure-openai",
        "name": "Azure OpenAI",
        "models": [
            {
                "id": "gpt-4o",
                "name": "GPT-4o (Azure)",
                "match": {"or": [{"starts_with": "gpt-4o"}]},
                "prices": {"input_mtok": 2.6, "output_mtok": 10.5},
                "context_window": 128000,
                "prices_checked": "2026-01-01",
            }
        ],
    },
]


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


def test_pricing_direct_model_match(client: TestClient) -> None:
    response = client.get("/pricing", params={"model": "gpt-4o"})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "gpt-4o"
    assert len(body["providers"]) == 2
    provider_ids = {p["provider"] for p in body["providers"]}
    assert provider_ids == {"openai", "azure-openai"}
    assert all(p["input_price_per_million"] is not None for p in body["providers"])


def test_pricing_provider_filter_returns_single(client: TestClient) -> None:
    response = client.get(
        "/pricing",
        params={"model": "gpt-4o", "provider": "openai"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "openai"
    assert body["input_price_per_million"] == 2.5
    assert body["output_price_per_million"] == 10.0


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


def test_pricing_provider_matches_display_name(client: TestClient) -> None:
    response = client.get(
        "/pricing",
        params={"model": "claude-3-5-sonnet", "provider": "anthropic"},
    )
    assert response.status_code == 200
    assert response.json()["provider"] == "anthropic"


def test_pricing_alias_equals_match(client: TestClient) -> None:
    response = client.get(
        "/pricing",
        params={"model": "claude-3.5-sonnet", "provider": "anthropic"},
    )
    assert response.status_code == 200
    assert response.json()["model"] == "claude-3-5-sonnet"


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
        lambda *args, **kwargs: FakePricingResponse({"id": "openai"}),
    )
    assert load_pricing_data() == {"id": "openai"}


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
    assert results[0]["input_price_per_million"] == 0.5


def test_find_model_pricing_alias_equals_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lib.pricing.load_pricing_data", lambda: MOCK_DATA)
    results = find_model_pricing("claude-3.5-sonnet")
    assert len(results) == 1
    assert results[0]["model"] == "claude-3-5-sonnet"


def test_find_model_pricing_no_match_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lib.pricing.load_pricing_data", lambda: MOCK_DATA)
    assert find_model_pricing("unknown-model") == []
