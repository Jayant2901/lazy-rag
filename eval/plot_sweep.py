import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

COLORS = ["tab:blue", "tab:green", "tab:red", "tab:purple", "tab:brown"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", default=["data/sweep_results.json"])
    parser.add_argument("--labels", nargs="+", default=None,
                         help="One label per --results file; defaults to the filename stem")
    parser.add_argument("--out", default="data/sweep_plot.png")
    args = parser.parse_args()

    labels = args.labels or [Path(p).stem for p in args.results]
    if len(labels) != len(args.results):
        raise ValueError("--labels must have the same length as --results")

    fig, ax1 = plt.subplots(figsize=(7.5, 5.5))
    ax1.set_xlabel("confidence threshold")
    ax1.set_ylabel("exact match")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.set_ylabel("retrieval rate")
    ax2.set_ylim(0, 1)

    for i, (path, label) in enumerate(zip(args.results, labels)):
        with open(path, encoding="utf-8") as f:
            results = json.load(f)

        color = COLORS[i % len(COLORS)]
        thresholds = [r["threshold"] for r in results]
        em = [r["em"] for r in results]
        retrieval_rate = [r["retrieval_rate"] for r in results]

        ax1.plot(thresholds, em, marker="o", color=color, label=f"{label} EM")
        if "em_ci_low" in results[0]:
            lo = [r["em_ci_low"] for r in results]
            hi = [r["em_ci_high"] for r in results]
            ax1.fill_between(thresholds, lo, hi, color=color, alpha=0.15)

        ax2.plot(thresholds, retrieval_rate, marker="s", linestyle="--", color=color,
                  alpha=0.6, label=f"{label} retrieval rate")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=8)

    plt.title("lazy-rag: quality vs. retrieval rate by confidence threshold\n"
              "(solid = EM, shaded = 95% bootstrap CI, dashed = retrieval rate)")
    fig.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Wrote plot to {args.out}")


if __name__ == "__main__":
    main()
