"""Token usage cost computation using OpenRouter pricing."""

from __future__ import annotations

from typing import Literal, TypedDict

from lib.llm import TokenUsage
from lib.pricing import find_model_pricing


class CostReport(TypedDict):
    """Cost breakdown for a set of tokens at a model's OpenRouter prices."""

    model: str
    platform: str | None
    found: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_price_per_token: float | None
    output_price_per_token: float | None
    input_price_per_m: float | None
    output_price_per_m: float | None
    prompt_tokens_cost: float
    completion_tokens_cost: float
    cost: float
    currency: Literal["USD"]
    source: str


def no_pricing_report(model: str, platform: str | None, usage: TokenUsage) -> CostReport:
    """Build a :class:`CostReport` that carries no pricing information.

    The report has ``found`` set to ``False``, ``None`` prices, zero costs, and an
    empty ``source``.

    Args:
        model: The model identifier the tokens were consumed by.
        platform: The platform that served the request, if known.
        usage: The token usage to report.

    Returns:
        A zero-cost :class:`CostReport`.
    """
    return CostReport(
        model=model,
        platform=platform,
        found=False,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        input_price_per_token=None,
        output_price_per_token=None,
        input_price_per_m=None,
        output_price_per_m=None,
        prompt_tokens_cost=0.0,
        completion_tokens_cost=0.0,
        cost=0.0,
        currency="USD",
        source="",
    )


def cost_report(model: str, platform: str | None, usage: TokenUsage) -> CostReport:
    """Return a cost breakdown for ``usage`` at ``model``'s OpenRouter prices.

    The model's pricing is looked up through :func:`lib.pricing.find_model_pricing`,
    the same source as the ``/pricing`` endpoint. When the model has no usable
    OpenRouter pricing, the report carries ``found`` as ``False``, ``None`` prices,
    zero costs, and an empty ``source``.

    Args:
        model: The model identifier the tokens were consumed by.
        platform: The platform that served the request, if known.
        usage: The token usage to price.

    Returns:
        A :class:`CostReport` with per-token and per-million-token prices and the
        input, output, and total costs in USD.

    Raises:
        httpx.HTTPError: If the OpenRouter pricing data cannot be fetched.
    """
    prices = find_model_pricing(model, platform)

    if not prices and platform is not None:
        prices = find_model_pricing(model)

    if not prices:
        return no_pricing_report(model, platform, usage)

    pricing = prices[0]

    input_price_per_token = pricing["input_price_per_token"]
    output_price_per_token = pricing["output_price_per_token"]

    if input_price_per_token is None or output_price_per_token is None:
        return no_pricing_report(model, platform, usage)

    return CostReport(
        model=pricing["model"],
        platform=platform,
        found=True,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        input_price_per_token=input_price_per_token,
        output_price_per_token=output_price_per_token,
        input_price_per_m=round(input_price_per_token * 1_000_000, 6),
        output_price_per_m=round(output_price_per_token * 1_000_000, 6),
        prompt_tokens_cost=round(usage.prompt_tokens * input_price_per_token, 6),
        completion_tokens_cost=round(usage.completion_tokens * output_price_per_token, 6),
        cost=round(
            usage.prompt_tokens * input_price_per_token
            + usage.completion_tokens * output_price_per_token,
            6,
        ),
        currency="USD",
        source=pricing["source"],
    )
