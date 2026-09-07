import argparse
import json

from src.retriever import Retriever
from eval.run_eval import load_jsonl
from eval.threshold_sweep import precompute
from eval.metrics import exact_match


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--qa", default="data/qa_subset.jsonl")
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument("--out", default="data/overconfident_failures.json")
    args = parser.parse_args()

    retriever = Retriever(args.corpus)
    qa_set = load_jsonl(args.qa)
    draws = precompute(qa_set, retriever)

    failures = []
    for item, d in zip(qa_set, draws):
        if d.confidence < args.min_confidence:
            continue
        if exact_match(d.draft_answer, d.gold_answer):
            continue
        failures.append({
            "question": item["question"],
            "gold_answer": d.gold_answer,
            "model_answer_no_retrieval": d.draft_answer,
            "self_reported_confidence": d.confidence,
            "rag_answer": d.rag_result.answer,
            "rag_correct": bool(exact_match(d.rag_result.answer, d.gold_answer)),
        })

    failures.sort(key=lambda x: x["self_reported_confidence"], reverse=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    print(f"{len(failures)} / {len(qa_set)} questions: confidence >= {args.min_confidence} but wrong\n")
    for item in failures[:10]:
        print(f"[conf={item['self_reported_confidence']:.2f}] {item['question']}")
        print(f"  gold: {item['gold_answer']!r}  model said: {item['model_answer_no_retrieval']!r}")
        print(f"  with retrieval: {item['rag_answer']!r} (correct={item['rag_correct']})\n")

    print(f"Wrote {len(failures)} cases to {args.out}")


if __name__ == "__main__":
    main()
