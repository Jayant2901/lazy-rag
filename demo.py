"""30-second demo of the lazy-rag gate: shows confidence, the gate's decision, and the answer.

Usage: python -m demo "your question" [--threshold 0.6]
"""
import argparse

from src.confidence import answer_with_confidence
from src.pipeline import answer_with_context
from src.retriever import Retriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()

    draft_answer, confidence = answer_with_confidence(args.question)
    print(f"Self-reported confidence: {confidence:.2f} (threshold: {args.threshold:.2f})")

    if confidence >= args.threshold:
        print("Gate decision: SKIP retrieval (confident enough)")
        print(f"Answer: {draft_answer}")
    else:
        print("Gate decision: RETRIEVE (not confident enough)")
        retriever = Retriever(args.corpus)
        result = answer_with_context(args.question, retriever)
        print(f"Answer: {result.answer}")


if __name__ == "__main__":
    main()
