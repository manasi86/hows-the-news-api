"""Client for the FreeLLM OpenAI-compatible chat completions API."""

from dataclasses import dataclass
import json
from typing import Any, TypeAlias, cast

import httpx

from lib.config import (
    LLM_INPUT_PRICE_PER_MTOKEN,
    LLM_OUTPUT_PRICE_PER_MTOKEN,
    Settings,
    get_settings,
)

JsonObject: TypeAlias = dict[str, Any]


class LLMError(RuntimeError):
    """Error raised when an LLM request or response cannot be processed."""


@dataclass(frozen=True)
class TokenUsage:
    """Token usage reported by the LLM chat completions API."""

    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        """Return the total number of tokens consumed."""
        return self.prompt_tokens + self.completion_tokens

    def prompt_tokens_cost(self) -> float:
        """Return the cost in USD of the prompt tokens."""
        return round(self.prompt_tokens / 1_000_000 * LLM_INPUT_PRICE_PER_MTOKEN, 6)

    def completion_tokens_cost(self) -> float:
        """Return the cost in USD of the completion tokens."""
        return round(self.completion_tokens / 1_000_000 * LLM_OUTPUT_PRICE_PER_MTOKEN, 6)

    def cost(self) -> float:
        """Return the total cost in USD of the consumed tokens.

        Cost is the sum of the prompt and completion token costs, each computed
        using the configured per-million-token prices. The total is rounded so
        it always equals ``prompt_tokens_cost() + completion_tokens_cost()``.
        """
        return round(self.prompt_tokens_cost() + self.completion_tokens_cost(), 6)


@dataclass(frozen=True)
class ChatResult:
    """Parsed JSON response from the LLM along with the token usage."""

    data: JsonObject
    usage: TokenUsage


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


def _chat(settings: Settings, system_prompt: str, user_text: str) -> ChatResult:
    """Send a chat completion request and return the parsed JSON response.

    Raises:
        LLMError: If the request fails or the response is malformed.

    Returns:
        The parsed JSON object from the model response along with token usage.
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
    usage_data = data.get("usage") or {}
    usage = TokenUsage(
        prompt_tokens=int(usage_data.get("prompt_tokens", 0)),
        completion_tokens=int(usage_data.get("completion_tokens", 0)),
    )
    return ChatResult(data=_parse_json(content), usage=usage)


def chat(system_prompt: str, user_text: str) -> ChatResult:
    """Run a chat completion against FreeLLM and return parsed JSON output.

    Args:
        system_prompt: The system prompt (with guardrails) to use.
        user_text: The user-provided text to analyze.

    Returns:
        A :class:`ChatResult` containing the parsed JSON object from the model
        response and the token usage.

    Raises:
        MissingApiKeyError: If ``LLM_API_KEY`` is not configured.
        LLMError: If the request fails or the response is malformed.
    """
    return _chat(get_settings(), system_prompt, user_text)
