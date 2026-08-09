"""Router for the ``/cost`` endpoint."""

from fastapi import APIRouter, HTTPException, Query
import httpx

from api.schemas import CostRequest, CostResponse
from lib.cost import cost_report
from lib.llm import TokenUsage

router = APIRouter()


def _cost_response(
    model: str, platform: str | None, prompt_tokens: int, completion_tokens: int
) -> CostResponse:
    """Build a :class:`CostResponse` for the given model and token usage.

    Raises:
        HTTPException: 503 if the pricing data cannot be fetched, 404 if no
            pricing is available for the model.
    """
    usage = TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    try:
        report = cost_report(model, platform, usage)
    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=503, detail=f"Unable to fetch pricing data: {error}"
        ) from error
    if not report["found"]:
        raise HTTPException(
            status_code=404,
            detail={"message": "Pricing not found", "model": model, "platform": platform},
        )
    return CostResponse(**report)


@router.get("/cost", response_model=CostResponse)
def get_cost(
    model: str = Query(min_length=1),
    prompt_tokens: int = Query(ge=0),
    completion_tokens: int = Query(ge=0),
    platform: str | None = Query(default=None),
) -> CostResponse:
    """Return the cost in USD of the given token usage for ``model``.

    Prices come from the same OpenRouter data as the ``/pricing`` endpoint, with
    ``platform`` used to pick the closest matching provider.

    Args:
        model: The model identifier the tokens were consumed by.
        prompt_tokens: The number of input (prompt) tokens used.
        completion_tokens: The number of output (completion) tokens used.
        platform: The platform that served the request, if known.

    Returns:
        The input, output, and total costs along with the prices used.
    """
    return _cost_response(model, platform, prompt_tokens, completion_tokens)


@router.post("/cost", response_model=CostResponse)
def post_cost(request: CostRequest) -> CostResponse:
    """Return the cost in USD of the given token usage for ``model``.

    Prices come from the same OpenRouter data as the ``/pricing`` endpoint, with
    ``platform`` used to pick the closest matching provider.

    Args:
        request: The model and token usage to price.

    Returns:
        The input, output, and total costs along with the prices used.
    """
    return _cost_response(
        request.model, request.platform, request.prompt_tokens, request.completion_tokens
    )
