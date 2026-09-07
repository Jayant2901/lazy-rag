import argparse
import json

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="data/sweep_results.json")
    parser.add_argument("--out", default="data/sweep_plot.png")
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as f:
        results = json.load(f)

    thresholds = [r["threshold"] for r in results]
    em = [r["em"] for r in results]
    retrieval_rate = [r["retrieval_rate"] for r in results]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.set_xlabel("confidence threshold")
    ax1.set_ylabel("exact match", color="tab:blue")
    ax1.plot(thresholds, em, marker="o", color="tab:blue", label="EM")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("retrieval rate", color="tab:orange")
    ax2.plot(thresholds, retrieval_rate, marker="s", color="tab:orange", label="retrieval rate")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(0, 1)

    plt.title("lazy-rag: quality vs. retrieval rate by confidence threshold")
    fig.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Wrote plot to {args.out}")


if __name__ == "__main__":
    main()
