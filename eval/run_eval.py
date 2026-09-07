import argparse
import json
from tqdm import tqdm

from src.retriever import Retriever
from src.pipeline import no_rag, always_rag, lazy_rag
from eval.metrics import exact_match, f1

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

    print(f"{'pipeline':<12} {'EM':>6} {'F1':>6} {'retrieval_rate':>15}")
    for name in args.pipelines:
        run = PIPELINES[name]
        em_total, f1_total, retrieved_count = 0.0, 0.0, 0
        for item in tqdm(qa_set, desc=name):
            result = run(item["question"], retriever, args.threshold)
            em_total += exact_match(result.answer, item["answer"])
            f1_total += f1(result.answer, item["answer"])
            retrieved_count += int(result.retrieved)
        n = len(qa_set)
        print(f"{name:<12} {em_total / n:>6.3f} {f1_total / n:>6.3f} {retrieved_count / n:>15.3f}")


if __name__ == "__main__":
    main()
