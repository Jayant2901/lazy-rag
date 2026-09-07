import argparse
import json
from dataclasses import dataclass

from tqdm import tqdm

from src.retriever import Retriever
from src.pipeline import answer_with_context, PipelineResult
from src.confidence import answer_with_confidence
from eval.metrics import exact_match, f1, bootstrap_ci
from eval.run_eval import load_jsonl


@dataclass
class QuestionDraws:
    gold_answer: str
    draft_answer: str
    confidence: float
    rag_result: PipelineResult


def precompute(qa_set: list[dict], retriever: Retriever, k: int = 3) -> list[QuestionDraws]:
    """One confidence draw + one RAG generation per question, reused across every threshold."""
    draws = []
    for item in tqdm(qa_set, desc="precompute"):
        draft_answer, confidence = answer_with_confidence(item["question"])
        rag_result = answer_with_context(item["question"], retriever, k=k)
        draws.append(QuestionDraws(item["answer"], draft_answer, confidence, rag_result))
    return draws


def score_at_threshold(draws: list[QuestionDraws], threshold: float) -> dict:
    em_scores, f1_scores, retrieved_count = [], [], 0
    for d in draws:
        if d.confidence >= threshold:
            answer, retrieved = d.draft_answer, False
        else:
            answer, retrieved = d.rag_result.answer, True
        em_scores.append(exact_match(answer, d.gold_answer))
        f1_scores.append(f1(answer, d.gold_answer))
        retrieved_count += int(retrieved)
    n = len(draws)
    em_point, em_lo, em_hi = bootstrap_ci(em_scores)
    f1_point, f1_lo, f1_hi = bootstrap_ci(f1_scores)
    return {
        "threshold": threshold,
        "em": em_point,
        "em_ci_low": em_lo,
        "em_ci_high": em_hi,
        "f1": f1_point,
        "f1_ci_low": f1_lo,
        "f1_ci_high": f1_hi,
        "retrieval_rate": retrieved_count / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--qa", default="data/qa_subset.jsonl")
    parser.add_argument("--thresholds", default="0.0,0.3,0.5,0.6,0.7,0.8,0.9,0.95,1.0")
    parser.add_argument("--out", default="data/sweep_results.json")
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    retriever = Retriever(args.corpus)
    qa_set = load_jsonl(args.qa)

    draws = precompute(qa_set, retriever)

    results = []
    print(f"{'threshold':>10} {'EM':>6} {'F1':>6} {'retrieval_rate':>15}")
    for threshold in thresholds:
        row = score_at_threshold(draws, threshold)
        results.append(row)
        print(f"{row['threshold']:>10.2f} {row['em']:>6.3f} {row['f1']:>6.3f} {row['retrieval_rate']:>15.3f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote sweep results to {args.out}")


if __name__ == "__main__":
    main()
