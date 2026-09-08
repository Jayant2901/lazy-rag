import argparse
import json
from tqdm import tqdm

from src.retriever import Retriever
from src.pipeline import no_rag, always_rag, lazy_rag, lazy_rag_consistency
from eval.metrics import exact_match, f1, bootstrap_ci
from eval.significance import sign_test

PIPELINES = {
    "no-rag": lambda q, r, threshold: no_rag(q, r),
    "always-rag": lambda q, r, threshold: always_rag(q, r),
    "lazy-rag": lambda q, r, threshold: lazy_rag(q, r, threshold=threshold),
    "lazy-rag-consistency": lambda q, r, threshold: lazy_rag_consistency(q, r, threshold=threshold),
}
# lazy-rag-consistency samples 3x per question, so it's opt-in via --pipelines rather than run by default
DEFAULT_PIPELINES = ["no-rag", "always-rag", "lazy-rag"]


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--qa", default="data/qa.jsonl")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--pipelines", nargs="+", default=DEFAULT_PIPELINES, choices=list(PIPELINES.keys()))
    parser.add_argument("--out", default=None, help="path to write full results as JSON")
    args = parser.parse_args()

    retriever = Retriever(args.corpus)
    qa_set = load_jsonl(args.qa)

    em_scores: dict[str, list[float]] = {}
    f1_scores: dict[str, list[float]] = {}
    retrieval_rates: dict[str, float] = {}
    summary: dict[str, dict] = {}

    print(f"{'pipeline':<12} {'EM':>6} {'EM 95% CI':>16} {'F1':>6} {'retrieval_rate':>15}")
    for name in args.pipelines:
        run = PIPELINES[name]
        pipeline_em, pipeline_f1, retrieved_count = [], [], 0
        for item in tqdm(qa_set, desc=name):
            result = run(item["question"], retriever, args.threshold)
            pipeline_em.append(exact_match(result.answer, item["answer"]))
            pipeline_f1.append(f1(result.answer, item["answer"]))
            retrieved_count += int(result.retrieved)
        n = len(qa_set)
        em_scores[name] = pipeline_em
        f1_scores[name] = pipeline_f1
        retrieval_rates[name] = retrieved_count / n

        em_point, em_lo, em_hi = bootstrap_ci(pipeline_em)
        f1_point, f1_lo, f1_hi = bootstrap_ci(pipeline_f1)
        print(f"{name:<12} {em_point:>6.3f} {f'[{em_lo:.2f}, {em_hi:.2f}]':>16} "
              f"{f1_point:>6.3f} {retrieved_count / n:>15.3f}")
        summary[name] = {
            "em": em_point, "em_ci_low": em_lo, "em_ci_high": em_hi,
            "f1": f1_point, "f1_ci_low": f1_lo, "f1_ci_high": f1_hi,
            "retrieval_rate": retrieved_count / n, "n": n,
        }

    significance = {}
    for variant in ("lazy-rag", "lazy-rag-consistency"):
        if variant not in em_scores:
            continue
        print()
        for baseline in ("always-rag", "no-rag"):
            if baseline not in em_scores or baseline == variant:
                continue
            result = sign_test(em_scores[variant], em_scores[baseline])
            significance[f"{variant}_vs_{baseline}"] = result
            print(f"{variant} vs {baseline}: p={result['p_value']:.3f} "
                  f"({result['n_discordant']} discordant pairs, "
                  f"{variant} won {result['a_wins']}, {baseline} won {result['b_wins']})")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({
                "threshold": args.threshold,
                "n": len(qa_set),
                "pipelines": summary,
                "significance": significance,
            }, f, indent=2)
        print(f"\nWrote results to {args.out}")


if __name__ == "__main__":
    main()
