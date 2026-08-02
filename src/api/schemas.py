"""Pydantic request and response models for the API."""

from pydantic import BaseModel, Field, model_validator

from lib.config import MAX_TEXT_LENGTH
from lib.sentiment import Sentiment


class TextRequest(BaseModel):
    """Request body containing the text to process."""

    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)


class SummarizeRequest(BaseModel):
    """Request body for the ``/summarize`` endpoint, accepting text or a URL."""

    text: str | None = Field(default=None, min_length=1, max_length=MAX_TEXT_LENGTH)
    url: str | None = None

    @model_validator(mode="after")
    def _exactly_one_input(self) -> "SummarizeRequest":
        """Ensure exactly one of ``text`` or ``url`` is provided."""
        if (self.text is None) == (self.url is None):
            raise ValueError("Provide exactly one of 'text' or 'url'")
        return self


class SummarizeResponse(BaseModel):
    """Response from the ``/summarize`` endpoint."""

    is_news: bool
    summary: str | None = None
    reason: str = ""


class AnalyseResponse(BaseModel):
    """Response from the ``/analyse`` endpoint."""

    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
