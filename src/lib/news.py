"""News content detection and summarization via the LLM."""

from typing import TypedDict

from lib.article import fetch_article
from lib.config import MAX_TEXT_LENGTH
from lib.llm import chat
from lib.prompts import SUMMARIZE_SYSTEM_PROMPT


class SummaryResult(TypedDict):
    """Result of the news detection and summarization."""

    is_news: bool
    summary: str | None
    reason: str


def summarize(text: str | None = None, url: str | None = None) -> SummaryResult:
    """Check whether the input is news content and, if so, summarize it.

    Exactly one of ``text`` or ``url`` must be provided. When ``url`` is given,
    the article is fetched and extracted first; if extraction fails, the result
    reports ``is_news`` as ``False`` with an explanatory ``reason``.

    Args:
        text: The input text to analyze.
        url: A URL to a news article to fetch and analyze instead of ``text``.

    Returns:
        A :class:`SummaryResult` containing the news flag, summary, and reason.

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
            )
        if content is None:
            return SummaryResult(
                is_news=False,
                summary=None,
                reason="Could not extract the news article from the URL",
            )
        text = content[:MAX_TEXT_LENGTH]
    if text is None:
        raise ValueError("Provide exactly one of 'text' or 'url'")
    data = chat(SUMMARIZE_SYSTEM_PROMPT, text)
    return SummaryResult(
        is_news=bool(data.get("is_news")),
        summary=data.get("summary"),
        reason=str(data.get("reason", "")),
    )
