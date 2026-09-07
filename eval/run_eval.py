import argparse
import json
from tqdm import tqdm

from src.retriever import Retriever
from src.pipeline import no_rag, always_rag, lazy_rag
from eval.metrics import exact_match, f1, bootstrap_ci
from eval.significance import sign_test

PIPELINES = {
    "no-rag": lambda q, r, threshold: no_rag(q, r),
    "always-rag": lambda q, r, threshold: always_rag(q, r),
    "lazy-rag": lambda q, r, threshold: lazy_rag(q, r, threshold=threshold),
}


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--qa", default="data/qa.jsonl")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--pipelines", nargs="+", default=list(PIPELINES.keys()))
    args = parser.parse_args()

    retriever = Retriever(args.corpus)
    qa_set = load_jsonl(args.qa)

    em_scores: dict[str, list[float]] = {}
    retrieval_rates: dict[str, float] = {}

    print(f"{'pipeline':<12} {'EM':>6} {'EM 95% CI':>16} {'F1':>6} {'retrieval_rate':>15}")
    for name in args.pipelines:
        run = PIPELINES[name]
        pipeline_em, f1_total, retrieved_count = [], 0.0, 0
        for item in tqdm(qa_set, desc=name):
            result = run(item["question"], retriever, args.threshold)
            pipeline_em.append(exact_match(result.answer, item["answer"]))
            f1_total += f1(result.answer, item["answer"])
            retrieved_count += int(result.retrieved)
        n = len(qa_set)
        em_scores[name] = pipeline_em
        retrieval_rates[name] = retrieved_count / n

        em_point, em_lo, em_hi = bootstrap_ci(pipeline_em)
        print(f"{name:<12} {em_point:>6.3f} {f'[{em_lo:.2f}, {em_hi:.2f}]':>16} "
              f"{f1_total / n:>6.3f} {retrieved_count / n:>15.3f}")

    if "lazy-rag" in em_scores:
        print()
        for baseline in ("always-rag", "no-rag"):
            if baseline not in em_scores:
                continue
            result = sign_test(em_scores["lazy-rag"], em_scores[baseline])
            print(f"lazy-rag vs {baseline}: p={result['p_value']:.3f} "
                  f"({result['n_discordant']} discordant pairs, "
                  f"lazy-rag won {result['a_wins']}, {baseline} won {result['b_wins']})")


if __name__ == "__main__":
    main()
