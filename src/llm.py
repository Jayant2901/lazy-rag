import anthropic
from src.config import ANTHROPIC_API_KEY, GEN_MODEL

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def generate(prompt: str, max_tokens: int = 512, system: str | None = None) -> str:
    kwargs = {"model": GEN_MODEL, "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": prompt}]}
    if system:
        kwargs["system"] = system
    response = _client.messages.create(**kwargs)
    return response.content[0].text.strip()
