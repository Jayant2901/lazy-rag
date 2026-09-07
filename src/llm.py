from groq import Groq
from src.config import GROQ_API_KEY, GEN_MODEL

_client = Groq(api_key=GROQ_API_KEY)


def generate(prompt: str, max_tokens: int = 512, system: str | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = _client.chat.completions.create(
        model=GEN_MODEL, max_tokens=max_tokens, messages=messages
    )
    return response.choices[0].message.content.strip()
