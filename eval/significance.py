import math


def sign_test(scores_a: list[float], scores_b: list[float]) -> dict:
    """Exact two-sided sign test (binomial, p=0.5) on paired per-question EM outcomes.

    Equivalent to McNemar's test on the discordant-pair count, without pulling in
    scipy/statsmodels as a dependency. Only questions where the two pipelines
    disagree (one right, one wrong) carry information; ties are dropped.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must be paired (same length, same question order)")

    a_wins = sum(1 for a, b in zip(scores_a, scores_b) if a > b)
    b_wins = sum(1 for a, b in zip(scores_a, scores_b) if b > a)
    n = a_wins + b_wins

    if n == 0:
        return {"a_wins": a_wins, "b_wins": b_wins, "n_discordant": 0, "p_value": 1.0}

    k = min(a_wins, b_wins)
    p_one_side = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    p_value = min(1.0, 2 * p_one_side)
    return {"a_wins": a_wins, "b_wins": b_wins, "n_discordant": n, "p_value": p_value}
