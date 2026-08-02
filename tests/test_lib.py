import json

import httpx
import pytest

from lib.config import MissingApiKeyError, get_settings
from lib.llm import LLMError, _extract_json, _parse_json, chat


class FakeResponse:
    """Minimal stand-in for an ``httpx.Response`` used in tests."""

    def __init__(
        self,
        *,
        json_data: object | None = None,
        json_error: json.JSONDecodeError | None = None,
        status_error: httpx.HTTPStatusError | None = None,
    ) -> None:
        self._json_data = json_data
        self._json_error = json_error
        self._status_error = status_error
        self.status_code = 500 if status_error is not None else 200

    def raise_for_status(self) -> None:
        if self._status_error is not None:
            raise self._status_error

    def json(self) -> object:
        if self._json_error is not None:
            raise self._json_error
        assert self._json_data is not None
        return self._json_data


def _http_status_error() -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://llm.nalits.com/v1/chat/completions")
    response = httpx.Response(500, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


def _fake_post(
    *,
    json_data: object | None = None,
    json_error: json.JSONDecodeError | None = None,
    status_error: httpx.HTTPStatusError | None = None,
    request_error: httpx.RequestError | None = None,
) -> object:
    if request_error is not None:
        raise request_error
    return FakeResponse(json_data=json_data, json_error=json_error, status_error=status_error)


def _set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.nalits.com/v1")
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    monkeypatch.setenv("LLM_TIMEOUT", "30")


def _ok_response(content: str) -> object:
    return {"choices": [{"message": {"content": content}}]}


def test_get_settings_with_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    settings = get_settings()
    assert settings.llm_api_key == "test-key"
    assert settings.llm_base_url == "https://llm.nalits.com/v1"
    assert settings.llm_model == "auto"
    assert settings.llm_timeout == 30.0


def test_get_settings_with_custom_values(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch)
    settings = get_settings()
    assert settings.llm_api_key == "test-key"
    assert settings.llm_base_url == "https://llm.nalits.com/v1"
    assert settings.llm_model == "custom-model"
    assert settings.llm_timeout == 30.0


def test_get_settings_strips_trailing_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.nalits.com/v1/")
    assert get_settings().llm_base_url == "https://llm.nalits.com/v1"


def test_get_settings_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError):
        get_settings()


def test_extract_json_plain() -> None:
    assert _extract_json('{"is_news": true}') == '{"is_news": true}'


def test_extract_json_fenced() -> None:
    text = '```json\n{"is_news": true}\n```'
    assert _extract_json(text) == '{"is_news": true}'


def test_extract_json_with_prose() -> None:
    text = 'Here is the result: {"is_news": true}. Hope this helps.'
    assert _extract_json(text) == '{"is_news": true}'


def test_extract_json_missing_open_brace() -> None:
    with pytest.raises(LLMError):
        _extract_json("no braces at all")


def test_extract_json_missing_close_brace() -> None:
    with pytest.raises(LLMError):
        _extract_json('{"is_news": true')


def test_extract_json_closed_before_open() -> None:
    with pytest.raises(LLMError):
        _extract_json('} {"is_news": true')


def test_parse_json_valid() -> None:
    assert _parse_json('{"is_news": true}') == {"is_news": True}


def test_parse_json_invalid() -> None:
    with pytest.raises(LLMError):
        _parse_json('{"is_news": true,}')


def test_chat_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch)
    content = '{"is_news": true, "summary": "s", "reason": "r"}'
    monkeypatch.setattr(
        "lib.llm.httpx.post",
        lambda *args, **kwargs: _fake_post(json_data=_ok_response(content)),
    )
    result = chat("system", "user text")
    assert result == {"is_news": True, "summary": "s", "reason": "r"}


def test_chat_http_status_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch)
    monkeypatch.setattr(
        "lib.llm.httpx.post",
        lambda *args, **kwargs: _fake_post(status_error=_http_status_error()),
    )
    with pytest.raises(LLMError):
        chat("system", "user text")


def test_chat_request_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch)
    request_error = httpx.ConnectError(
        "boom", request=httpx.Request("POST", "https://llm.nalits.com/v1/chat/completions")
    )
    monkeypatch.setattr(
        "lib.llm.httpx.post",
        lambda *args, **kwargs: _fake_post(request_error=request_error),
    )
    with pytest.raises(LLMError):
        chat("system", "user text")


def test_chat_malformed_response_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch)
    monkeypatch.setattr(
        "lib.llm.httpx.post",
        lambda *args, **kwargs: _fake_post(json_data={}),
    )
    with pytest.raises(LLMError):
        chat("system", "user text")


def test_chat_malformed_response_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch)
    json_error = json.JSONDecodeError("boom", "bad", 0)
    monkeypatch.setattr(
        "lib.llm.httpx.post",
        lambda *args, **kwargs: _fake_post(json_error=json_error),
    )
    with pytest.raises(LLMError):
        chat("system", "user text")


def test_chat_invalid_content_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch)
    monkeypatch.setattr(
        "lib.llm.httpx.post",
        lambda *args, **kwargs: _fake_post(json_data=_ok_response("not json")),
    )
    with pytest.raises(LLMError):
        chat("system", "user text")
