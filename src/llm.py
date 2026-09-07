import re
import time

import groq
from groq import Groq
from src.config import GROQ_API_KEY, GEN_MODEL

_client = Groq(api_key=GROQ_API_KEY)


def _retry_delay(error: groq.RateLimitError, attempt: int) -> float:
    message = str(error)
    match = re.search(r"try again in ([\d.]+)(ms|s)", message)
    if match:
        value, unit = match.groups()
        seconds = float(value) / 1000 if unit == "ms" else float(value)
        return max(seconds, 1.0)
    return min(2 ** attempt, 30)


def generate(prompt: str, max_tokens: int = 512, system: str | None = None, max_retries: int = 6) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(max_retries):
        try:
            response = _client.chat.completions.create(
                model=GEN_MODEL, max_tokens=max_tokens, messages=messages
            )
            return response.choices[0].message.content.strip()
        except groq.RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(_retry_delay(e, attempt))
