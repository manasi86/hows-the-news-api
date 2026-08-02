"""Application configuration loaded from environment variables and ``.env``."""

from dataclasses import dataclass
from os import environ

from dotenv import load_dotenv

load_dotenv()

LLM_BASE_URL_DEFAULT = "https://llm.nalits.com/v1"
LLM_MODEL_DEFAULT = "auto"
LLM_TIMEOUT_DEFAULT = 30.0

MAX_TEXT_LENGTH = 10_000


class MissingApiKeyError(RuntimeError):
    """Error raised when ``LLM_API_KEY`` is not configured."""


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the FreeLLM integration."""

    llm_api_key: str
    llm_base_url: str = LLM_BASE_URL_DEFAULT
    llm_model: str = LLM_MODEL_DEFAULT
    llm_timeout: float = LLM_TIMEOUT_DEFAULT


def get_settings() -> Settings:
    """Build a :class:`Settings` instance from the environment.

    Raises:
        MissingApiKeyError: If ``LLM_API_KEY`` is not set.

    Returns:
        Populated settings.
    """
    api_key = environ.get("LLM_API_KEY", "")
    if not api_key:
        raise MissingApiKeyError("LLM_API_KEY is not set in the environment or .env file")
    base_url = environ.get("LLM_BASE_URL", LLM_BASE_URL_DEFAULT).rstrip("/")
    model = environ.get("LLM_MODEL", LLM_MODEL_DEFAULT)
    timeout = float(environ.get("LLM_TIMEOUT", LLM_TIMEOUT_DEFAULT))
    return Settings(
        llm_api_key=api_key,
        llm_base_url=base_url,
        llm_model=model,
        llm_timeout=timeout,
    )
