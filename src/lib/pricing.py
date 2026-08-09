"""Model pricing lookups via the LLM Stats API with a web-search fallback."""

from __future__ import annotations

from typing import Any, TypedDict
import httpx

PRICING_URL = "https://raw.githubusercontent.com/pydantic/genai-prices/main/prices/data_slim.json"


class PricingResult(TypedDict):
    """Pricing details for a single model on a single provider."""

    model: str
    model_name: str | None
    provider: str
    input_price_per_million: float | None
    output_price_per_million: float | None
    currency: str
    context_window: int | None
    prices_checked: str | None
    source: str


def load_pricing_data() -> Any:
    response = httpx.get(PRICING_URL, timeout=10)
    response.raise_for_status()

    return response.json()


def normalise(value: str) -> str:
    return value.lower().strip().replace("_", "-")


def find_model_pricing(model_name: str, provider_name: str | None = None) -> list[PricingResult]:
    data = load_pricing_data()

    requested_model = normalise(model_name)

    requested_provider = normalise(provider_name) if provider_name else None

    results: list[PricingResult] = []

    # genai-prices contains provider definitions
    for provider in data:
        provider_id = normalise(provider.get("id", ""))

        provider_display_name = normalise(provider.get("name", ""))

        # Provider filter
        if requested_provider:
            provider_exact_match = requested_provider in (provider_id, provider_display_name)

            provider_fuzzy_match = (
                requested_provider in provider_id or requested_provider in provider_display_name
            )

            if not provider_exact_match and not provider_fuzzy_match:
                continue

        models = provider.get("models", [])

        for model in models:
            model_id = normalise(model.get("id", ""))

            model_display_name = normalise(model.get("name", ""))

            # Check direct model match
            direct_match = requested_model in (model_id, model_display_name)

            # Check aliases
            alias_match = False

            match_config = model.get("match", {})

            match_rules = match_config.get("or", [])

            for rule in match_rules:
                equals = rule.get("equals")

                if equals and normalise(equals) == requested_model:
                    alias_match = True
                    break

            if not direct_match and not alias_match:
                continue

            prices = model.get("prices", {})

            input_price = prices.get("input_mtok")

            output_price = prices.get("output_mtok")

            results.append(
                {
                    "model": model.get("id", model_name),
                    "model_name": model.get("name"),
                    "provider": provider.get("id", provider.get("name")),
                    "input_price_per_million": input_price,
                    "output_price_per_million": output_price,
                    "currency": "USD",
                    "context_window": model.get("context_window"),
                    "prices_checked": model.get("prices_checked"),
                    "source": "pydantic/genai-prices",
                }
            )

    # Prefer exact provider matches over substring matches
    if requested_provider:
        exact_matches = [
            result for result in results if requested_provider == normalise(result["provider"])
        ]

        if exact_matches:
            results = exact_matches

    return results
