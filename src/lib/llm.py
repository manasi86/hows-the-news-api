"""Client for the FreeLLM OpenAI-compatible chat completions API."""

import json
from typing import Any, TypeAlias, cast

import httpx

from lib.config import Settings, get_settings

JsonObject: TypeAlias = dict[str, Any]


class LLMError(RuntimeError):
    """Error raised when an LLM request or response cannot be processed."""


def _extract_json(response_text: str) -> str:
    """Extract the outermost JSON object from a model response.

    Model responses occasionally wrap JSON in markdown code fences or surround
    it with prose, so the JSON object is located and extracted directly.

    Raises:
        LLMError: If no JSON object can be found in the response.

    Returns:
        The extracted JSON text.
    """
    start = response_text.find("{")
    if start == -1:
        raise LLMError("No JSON object found in LLM response")
    end = response_text.rfind("}")
    if end == -1:
        raise LLMError("No JSON object found in LLM response")
    if end < start:
        raise LLMError("No JSON object found in LLM response")
    return response_text[start : end + 1]


def _parse_json(response_text: str) -> JsonObject:
    """Parse a model response as a JSON object.

    Raises:
        LLMError: If the response is not valid JSON.

    Returns:
        The parsed JSON object.
    """
    try:
        return cast(JsonObject, json.loads(_extract_json(response_text)))
    except json.JSONDecodeError as exc:
        raise LLMError("LLM response was not valid JSON") from exc


def _chat(settings: Settings, system_prompt: str, user_text: str) -> JsonObject:
    """Send a chat completion request and return the parsed JSON response.

    Raises:
        LLMError: If the request fails or the response is malformed.

    Returns:
        The parsed JSON object from the model response.
    """
    url = f"{settings.llm_base_url}/chat/completions"
    payload: JsonObject = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=settings.llm_timeout)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise LLMError(f"LLM request failed with HTTP status {exc.response.status_code}") from exc
    except httpx.RequestError as exc:
        raise LLMError("LLM request could not be completed") from exc
    try:
        data = response.json()
        content: str = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise LLMError("LLM response was malformed") from exc
    return _parse_json(content)


def chat(system_prompt: str, user_text: str) -> JsonObject:
    """Run a chat completion against FreeLLM and return parsed JSON output.

    Args:
        system_prompt: The system prompt (with guardrails) to use.
        user_text: The user-provided text to analyze.

    Returns:
        The parsed JSON object from the model response.

    Raises:
        MissingApiKeyError: If ``LLM_API_KEY`` is not configured.
        LLMError: If the request fails or the response is malformed.
    """
    return _chat(get_settings(), system_prompt, user_text)
