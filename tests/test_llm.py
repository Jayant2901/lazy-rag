import httpx
import groq
import pytest

from src.llm import _retry_delay

_RESPONSE = httpx.Response(status_code=429, request=httpx.Request("POST", "http://example.test"))


def _rate_limit_error(message: str) -> groq.RateLimitError:
    return groq.RateLimitError(message, response=_RESPONSE, body=None)


def test_retry_delay_parses_seconds():
    error = _rate_limit_error("Rate limit reached, please try again in 2.5s")
    assert _retry_delay(error, attempt=0) == pytest.approx(2.5)


def test_retry_delay_parses_milliseconds():
    error = _rate_limit_error("Rate limit reached, please try again in 1500ms")
    assert _retry_delay(error, attempt=0) == pytest.approx(1.5)


def test_retry_delay_parses_minutes_and_seconds():
    error = _rate_limit_error("Rate limit reached, please try again in 1m30s")
    assert _retry_delay(error, attempt=0) == pytest.approx(90.0)


def test_retry_delay_floors_at_one_second():
    error = _rate_limit_error("Rate limit reached, please try again in 0.01s")
    assert _retry_delay(error, attempt=0) == 1.0


def test_retry_delay_falls_back_to_exponential_backoff_when_unparseable():
    error = _rate_limit_error("Rate limit reached, no wait time given")
    assert _retry_delay(error, attempt=0) == 1.0
    assert _retry_delay(error, attempt=2) == 4.0


def test_retry_delay_backoff_caps_at_thirty_seconds():
    generic_error = groq.APIConnectionError(request=httpx.Request("POST", "http://example.test"))
    assert _retry_delay(generic_error, attempt=10) == 30
