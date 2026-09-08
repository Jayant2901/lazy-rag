from collections import Counter

from eval.metrics import normalize
from src.llm import generate


def consistency_score(samples: list[str]) -> tuple[str, float]:
    """Pure scoring logic, split out so it's testable without an LLM call.

    FLARE triggers retrieval from token-level output probability, which Groq
    does not expose (no model on the free tier supports `logprobs`). This is
    a black-box substitute in the same spirit: sample the same question
    multiple times and use answer agreement as the uncertainty signal
    (SelfCheckGPT-style self-consistency), instead of self-verbalized
    confidence or token probability. Returns the majority-vote answer and the
    fraction of samples agreeing with it.
    """
    normalized = [normalize(s) for s in samples]
    top_answer, top_count = Counter(normalized).most_common(1)[0]
    majority_sample = next(s for s, n in zip(samples, normalized) if n == top_answer)
    return majority_sample, top_count / len(samples)


def answer_with_consistency(question: str, n_samples: int = 3, temperature: float = 1.0) -> tuple[str, float]:
    samples = [generate(question, temperature=temperature) for _ in range(n_samples)]
    return consistency_score(samples)
