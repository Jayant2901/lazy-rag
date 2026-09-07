from eval.metrics import normalize, exact_match, f1, bootstrap_ci


def test_normalize_lowercases_and_strips_punctuation():
    assert normalize("The Catholic Church!") == "catholic church"


def test_normalize_strips_articles():
    assert normalize("a cat") == "cat"
    assert normalize("an apple") == "apple"
    assert normalize("the dog") == "dog"


def test_normalize_collapses_whitespace():
    assert normalize("  many   spaces  ") == "many spaces"


def test_exact_match_case_insensitive():
    assert exact_match("PARIS", "paris") == 1.0


def test_exact_match_ignores_articles_and_punctuation():
    assert exact_match("the Catholic Church.", "Catholic Church") == 1.0


def test_exact_match_empty_strings():
    assert exact_match("", "") == 1.0
    assert exact_match("", "Paris") == 0.0
    assert exact_match("Paris", "") == 0.0


def test_exact_match_mismatch():
    assert exact_match("London", "Paris") == 0.0


def test_f1_identical_strings():
    assert f1("Paris", "Paris") == 1.0


def test_f1_partial_token_overlap():
    # 1 shared token ("church") out of 2 pred tokens, 1 gold token -> P=0.5, R=1.0
    score = f1("catholic church", "church")
    assert 0.5 < score < 0.8


def test_f1_no_overlap():
    assert f1("London", "Paris") == 0.0


def test_f1_empty_prediction():
    assert f1("", "Paris") == 0.0


def test_f1_empty_both():
    assert f1("", "") == 1.0


def test_bootstrap_ci_empty_returns_zeros():
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)


def test_bootstrap_ci_constant_scores_has_zero_width_interval():
    point, lo, hi = bootstrap_ci([1.0] * 20, n_boot=200, seed=1)
    assert point == 1.0
    assert lo == 1.0
    assert hi == 1.0


def test_bootstrap_ci_bounds_contain_point_estimate():
    scores = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    point, lo, hi = bootstrap_ci(scores, n_boot=500, seed=0)
    assert lo <= point <= hi


def test_bootstrap_ci_is_deterministic_given_seed():
    scores = [1.0, 0.0, 1.0, 0.0, 1.0]
    result_a = bootstrap_ci(scores, seed=42)
    result_b = bootstrap_ci(scores, seed=42)
    assert result_a == result_b
