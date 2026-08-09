"""News content detection and summarization via the LLM."""

from typing import TypedDict

import httpx

from lib.article import fetch_article
from lib.config import MAX_TEXT_LENGTH
from lib.cost import cost_report, no_pricing_report
from lib.llm import chat
from lib.prompts import SUMMARIZE_SYSTEM_PROMPT


class SummaryResult(TypedDict):
    """Result of the news detection and summarization."""

    is_news: bool
    summary: str | None
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


def summarize(text: str | None = None, url: str | None = None) -> SummaryResult:
    """Check whether the input is news content and, if so, summarize it.

    Exactly one of ``text`` or ``url`` must be provided. When ``url`` is given,
    the article is fetched and extracted first; if extraction fails, the result
    reports ``is_news`` as ``False`` with an explanatory ``reason``.

    Args:
        text: The input text to analyze.
        url: A URL to a news article to fetch and analyze instead of ``text``.

    Returns:
        A :class:`SummaryResult` containing the news flag, summary, reason, and
        the LLM token usage and cost.

    Raises:
        ValueError: If neither ``text`` nor ``url`` is provided.
    """
    if url is not None:
        content, error = fetch_article(url)
        if error is not None:
            return SummaryResult(
                is_news=False,
                summary=None,
                reason=f"Could not extract the news article from the URL: {error}",
                model="",
                platform=None,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                prompt_tokens_cost=0.0,
                completion_tokens_cost=0.0,
                cost=0.0,
                input_price_per_token=None,
                output_price_per_token=None,
                input_price_per_m=None,
                output_price_per_m=None,
                source="",
            )
        if content is None:
            return SummaryResult(
                is_news=False,
                summary=None,
                reason="Could not extract the news article from the URL",
                model="",
                platform=None,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                prompt_tokens_cost=0.0,
                completion_tokens_cost=0.0,
                cost=0.0,
                input_price_per_token=None,
                output_price_per_token=None,
                input_price_per_m=None,
                output_price_per_m=None,
                source="",
            )
        text = content[:MAX_TEXT_LENGTH]
    if text is None:
        raise ValueError("Provide exactly one of 'text' or 'url'")
    result = chat(SUMMARIZE_SYSTEM_PROMPT, text)
    try:
        cost = cost_report(result.model, result.platform, result.usage)
    except httpx.HTTPError:
        cost = no_pricing_report(result.model, result.platform, result.usage)

    return SummaryResult(
        is_news=bool(result.data.get("is_news")),
        summary=result.data.get("summary"),
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
