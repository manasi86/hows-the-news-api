"""Router for the ``/analyse`` endpoint."""

from fastapi import APIRouter, HTTPException

from api.schemas import AnalyseResponse, TextRequest
from lib import sentiment
from lib.config import MissingApiKeyError
from lib.llm import LLMError

router = APIRouter()


@router.post("/analyse", response_model=AnalyseResponse)
def analyse(request: TextRequest) -> AnalyseResponse:
    """Analyze the sentiment of the input text summary.

    Args:
        request: The request containing the text to process.

    Returns:
        The sentiment analysis result.

    Raises:
        HTTPException: 502 if the LLM call fails, 500 if the API key is missing.
    """
    try:
        result = sentiment.analyze_sentiment(request.text)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except MissingApiKeyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AnalyseResponse(**result)
