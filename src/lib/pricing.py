"""Model pricing lookups via the OpenRouter models API."""

from __future__ import annotations

from functools import lru_cache
import logging
from typing import Any, TypedDict, cast

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


class PricingResult(TypedDict):
    """Pricing details for a single model served by OpenRouter."""

    model: str
    model_name: str | None
    provider: str
    input_price_per_token: float | None
    output_price_per_token: float | None
    currency: str
    context_window: int | None
    source: str


def load_pricing_data() -> Any:
    response = httpx.get(OPENROUTER_MODELS_URL, timeout=10)
    response.raise_for_status()

    return response.json()


def normalise(value: str) -> str:
    return value.lower().strip().replace("_", "-")


def _to_float(value: object) -> float | None:
    try:
        return float(cast("Any", value)) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=128)
def find_model_pricing(model_name: str, provider_name: str | None = None) -> list[PricingResult]:
    logger.info("data from internet")
    data = load_pricing_data()

    requested_model = normalise(model_name)

    requested_provider = normalise(provider_name) if provider_name else None

    results: list[PricingResult] = []

    # OpenRouter model ids are ``provider/model``, optionally with a leading
    # ``~`` featured marker or a trailing ``:variant`` suffix.
    for model in data.get("data", []):
        model_id = normalise(model.get("id", "")).lstrip("~")

        provider = model_id.split("/", 1)[0]

        # Provider filter
        if requested_provider and requested_provider not in provider:
            continue

        model_part = model_id.split("/", 1)[1] if "/" in model_id else model_id
        model_part_no_variant = model_part.split(":", 1)[0]

        model_display_name = normalise(model.get("name", ""))

        direct_match = requested_model == model_id

        name_match = requested_model == model_display_name

        bare_match = requested_model in (model_part, model_part_no_variant)

        if not direct_match and not name_match and not bare_match:
            continue

        pricing = model.get("pricing", {})

        results.append(
            {
                "model": model_id,
                "model_name": model.get("name"),
                "provider": provider,
                "input_price_per_token": _to_float(pricing.get("prompt")),
                "output_price_per_token": _to_float(pricing.get("completion")),
                "currency": "USD",
                "context_window": model.get("context_length"),
                "source": "openrouter",
            }
        )

    # Prefer exact provider matches over substring matches
    if requested_provider:
        exact_matches = [result for result in results if requested_provider == result["provider"]]

        if exact_matches:
            results = exact_matches

    return results
