"""Sentiment analysis of news summaries via the LLM."""

from typing import Literal, TypedDict, cast

from lib.llm import chat
from lib.prompts import SENTIMENT_SYSTEM_PROMPT

Sentiment = Literal["positive", "negative", "neutral"]


class SentimentResult(TypedDict):
    """Result of the sentiment analysis."""

    sentiment: Sentiment
    confidence: float
    reason: str


def analyze_sentiment(text: str) -> SentimentResult:
    """Classify the sentiment of ``text`` as positive, negative, or neutral.

    Args:
        text: The input text (typically a news summary) to analyze.

    Returns:
        A :class:`SentimentResult` containing the label, confidence, and reason.
    """
    data = chat(SENTIMENT_SYSTEM_PROMPT, text)
    return SentimentResult(
        sentiment=cast(Sentiment, data.get("sentiment", "neutral")),
        confidence=float(data.get("confidence", 0.0)),
        reason=str(data.get("reason", "")),
    )
