import re
import time

import groq
from groq import Groq
from src.config import GROQ_API_KEY, GEN_MODEL

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def _retry_delay(error: groq.RateLimitError, attempt: int) -> float:
    message = str(error)
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)(ms|s)\b", message)
    if match:
        minutes, value, unit = match.groups()
        seconds = float(value) / 1000 if unit == "ms" else float(value)
        seconds += float(minutes) * 60 if minutes else 0
        return max(seconds, 1.0)
    return min(2 ** attempt, 30)


def generate(
    prompt: str, max_tokens: int = 512, max_retries: int = 6, temperature: float | None = None
) -> str:
    messages = [{"role": "user", "content": prompt}]
    kwargs = {} if temperature is None else {"temperature": temperature}

    for attempt in range(max_retries):
        try:
            response = _get_client().chat.completions.create(
                model=GEN_MODEL, max_tokens=max_tokens, messages=messages, **kwargs
            )
            return response.choices[0].message.content.strip()
        except groq.RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(_retry_delay(e, attempt))
