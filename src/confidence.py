import re
from src.llm import generate

CONFIDENCE_PROMPT = """Answer the question as best you can from your own knowledge.
Then on a new line, rate your confidence in that answer from 0.0 (guessing) to 1.0 (certain).

Question: {question}

Format exactly as:
Answer: <your answer>
Confidence: <0.0-1.0>"""


def parse_confidence_response(raw: str) -> tuple[str, float]:
    """Pure parsing logic, split out so it's testable without an LLM call.

    Note: a missing/malformed Confidence line defaults to 0.0 (max uncertainty),
    which biases toward retrieval on unparseable output rather than skipping it -
    the safer failure direction for this project, but worth knowing about.
    """
    answer_match = re.search(r"Answer:\s*(.+)", raw)
    confidence_match = re.search(r"Confidence:\s*([\d.]+)", raw)
    answer = answer_match.group(1).strip() if answer_match else raw.strip()
    confidence = float(confidence_match.group(1)) if confidence_match else 0.0
    return answer, confidence


def answer_with_confidence(question: str) -> tuple[str, float]:
    raw = generate(CONFIDENCE_PROMPT.format(question=question), max_tokens=300)
    return parse_confidence_response(raw)
