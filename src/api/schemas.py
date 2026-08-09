"""Pydantic request and response models for the API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    model: str = ""
    platform: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_cost: float = 0.0
    completion_tokens_cost: float = 0.0
    cost: float = 0.0
    input_price_per_token: float | None = None
    output_price_per_token: float | None = None
    input_price_per_m: float | None = None
    output_price_per_m: float | None = None
    source: str = ""


class AnalyseResponse(BaseModel):
    """Response from the ``/analyse`` endpoint."""

    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    model: str = ""
    platform: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_cost: float = 0.0
    completion_tokens_cost: float = 0.0
    cost: float = 0.0
    input_price_per_token: float | None = None
    output_price_per_token: float | None = None
    input_price_per_m: float | None = None
    output_price_per_m: float | None = None
    source: str = ""


class PricingResponse(BaseModel):
    """Response from the ``/pricing`` endpoint."""

    model_config = ConfigDict(extra="forbid")

    model: str
    model_name: str | None = None
    provider: str | None = None
    input_price_per_token: float | None = None
    output_price_per_token: float | None = None
    currency: Literal["USD"] = "USD"
    context_window: int | None = None
    source: Literal["openrouter"] = "openrouter"


class PricingListResponse(BaseModel):
    """Response from the ``/pricing`` endpoint when no provider is given."""

    model: str
    providers: list[PricingResponse]


class CostRequest(BaseModel):
    """Request body for the ``/cost`` endpoint."""

    model: str = Field(min_length=1)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    platform: str | None = None


class CostResponse(BaseModel):
    """Response from the ``/cost`` endpoint."""

    model: str
    platform: str | None = None
    found: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    input_price_per_token: float | None = None
    output_price_per_token: float | None = None
    input_price_per_m: float | None = None
    output_price_per_m: float | None = None
    prompt_tokens_cost: float = 0.0
    completion_tokens_cost: float = 0.0
    cost: float = 0.0
    currency: Literal["USD"] = "USD"
    source: str = ""
