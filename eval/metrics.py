import re
import string
from collections import Counter

import numpy as np


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize(prediction) == normalize(gold))


def f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize(prediction).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def bootstrap_ci(
    scores: list[float], n_boot: int = 1000, ci: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile bootstrap CI on the mean of per-question scores. Returns (point, lower, upper)."""
    arr = np.array(scores, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    resample_idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[resample_idx].mean(axis=1)
    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_means, 100 * alpha))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return float(arr.mean()), lower, upper
