import argparse
import json

from src.retriever import Retriever
from src.pipeline import lazy_rag
from eval.metrics import exact_match, f1
from eval.run_eval import load_jsonl


def run_at_threshold(qa_set: list[dict], retriever: Retriever, threshold: float) -> dict:
    em_total, f1_total, retrieved_count = 0.0, 0.0, 0
    for item in qa_set:
        result = lazy_rag(item["question"], retriever, threshold=threshold)
        em_total += exact_match(result.answer, item["answer"])
        f1_total += f1(result.answer, item["answer"])
        retrieved_count += int(result.retrieved)
    n = len(qa_set)
    return {
        "threshold": threshold,
        "em": em_total / n,
        "f1": f1_total / n,
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

    results = []
    print(f"{'threshold':>10} {'EM':>6} {'F1':>6} {'retrieval_rate':>15}")
    for threshold in thresholds:
        row = run_at_threshold(qa_set, retriever, threshold)
        results.append(row)
        print(f"{row['threshold']:>10.2f} {row['em']:>6.3f} {row['f1']:>6.3f} {row['retrieval_rate']:>15.3f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote sweep results to {args.out}")


if __name__ == "__main__":
    main()
