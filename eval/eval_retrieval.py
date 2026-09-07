import argparse

from src.retriever import Retriever
from eval.run_eval import load_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--qa", default="data/qa.jsonl")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5])
    args = parser.parse_args()

    retriever = Retriever(args.corpus)
    qa_set = load_jsonl(args.qa)

    n_with_gold = sum(1 for item in qa_set if item.get("gold_title"))
    if n_with_gold == 0:
        print("No questions carry a 'gold_title' field - regenerate data with "
              "eval/prepare_popqa.py to enable Recall@k.")
        return

    print(f"Recall@k over {n_with_gold}/{len(qa_set)} questions with a gold document:\n")
    for k in args.k:
        recall = retriever.recall_at_k(qa_set, k=k)
        print(f"Recall@{k}: {recall:.3f}")


if __name__ == "__main__":
    main()
