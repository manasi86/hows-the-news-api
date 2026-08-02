"""Router for the ``/summarize`` endpoint."""

from fastapi import APIRouter, HTTPException

from api.schemas import SummarizeRequest, SummarizeResponse
from lib import news
from lib.config import MissingApiKeyError
from lib.llm import LLMError

router = APIRouter()


@router.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest) -> SummarizeResponse:
    """Check whether the input text or URL is news content and, if so, summarize it.

    If a URL is provided and its article cannot be fetched or extracted, the
    response reports ``is_news`` as ``False`` with an explanatory ``reason``.

    Args:
        request: The request containing the text or URL to process.

    Returns:
        The summarization result.

    Raises:
        HTTPException: 502 if the LLM call fails, 500 if the API key is missing.
    """
    try:
        result = news.summarize(request.text, request.url)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except MissingApiKeyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return SummarizeResponse(**result)
