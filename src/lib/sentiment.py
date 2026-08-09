"""Sentiment analysis of news summaries via the LLM."""

from typing import Literal, TypedDict, cast

import httpx

from lib.cost import cost_report, no_pricing_report
from lib.llm import chat
from lib.prompts import SENTIMENT_SYSTEM_PROMPT

Sentiment = Literal["positive", "negative", "neutral"]


class SentimentResult(TypedDict):
    """Result of the sentiment analysis."""

    sentiment: Sentiment
    confidence: float
    reason: str
    model: str
    platform: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_cost: float
    completion_tokens_cost: float
    cost: float
    input_price_per_token: float | None
    output_price_per_token: float | None
    input_price_per_m: float | None
    output_price_per_m: float | None
    source: str


def analyze_sentiment(text: str) -> SentimentResult:
    """Classify the sentiment of ``text`` as positive, negative, or neutral.

    Args:
        text: The input text (typically a news summary) to analyze.

    Returns:
        A :class:`SentimentResult` containing the label, confidence, reason, and
        the LLM token usage and cost.
    """
    result = chat(SENTIMENT_SYSTEM_PROMPT, text)
    try:
        cost = cost_report(result.model, result.platform, result.usage)
    except httpx.HTTPError:
        cost = no_pricing_report(result.model, result.platform, result.usage)
    return SentimentResult(
        sentiment=cast(Sentiment, result.data.get("sentiment", "neutral")),
        confidence=float(result.data.get("confidence", 0.0)),
        reason=str(result.data.get("reason", "")),
        model=result.model,
        platform=result.platform,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        total_tokens=result.usage.total_tokens,
        prompt_tokens_cost=cost["prompt_tokens_cost"],
        completion_tokens_cost=cost["completion_tokens_cost"],
        cost=cost["cost"],
        input_price_per_token=cost["input_price_per_token"],
        output_price_per_token=cost["output_price_per_token"],
        input_price_per_m=cost["input_price_per_m"],
        output_price_per_m=cost["output_price_per_m"],
        source=cost["source"],
    )
