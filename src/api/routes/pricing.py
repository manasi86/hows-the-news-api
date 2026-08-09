"""Router for the ``/pricing`` endpoint."""

from fastapi import APIRouter, HTTPException
import httpx

# from fastapi import Query
from api.schemas import PricingListResponse, PricingResponse
from lib.pricing import find_model_pricing

router = APIRouter()


@router.get("/pricing", response_model=PricingResponse | PricingListResponse)
def get_pricing(model: str, provider: str | None = None) -> PricingResponse | PricingListResponse:

    try:
        prices = find_model_pricing(model, provider)

    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=503, detail=f"Unable to fetch pricing data: {error}"
        ) from error

    if not prices:
        raise HTTPException(
            status_code=404,
            detail={"message": "Pricing not found", "model": model, "provider": provider},
        )

    # If provider specified, normally return one result
    if provider and len(prices) == 1:
        return PricingResponse.model_validate(prices[0])

    return PricingListResponse(
        model=model,
        providers=[PricingResponse.model_validate(price) for price in prices],
    )
