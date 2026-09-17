import pytest

from eval.significance import sign_test


def test_all_ties_gives_p_one():
    result = sign_test([1.0, 0.0, 1.0], [1.0, 0.0, 1.0])
    assert result == {"a_wins": 0, "b_wins": 0, "n_discordant": 0, "p_value": 1.0}


def test_lopsided_split_gives_tiny_p_value():
    a = [1.0] * 10
    b = [0.0] * 10
    result = sign_test(a, b)
    assert result["a_wins"] == 10
    assert result["b_wins"] == 0
    assert result["n_discordant"] == 10
    assert result["p_value"] == pytest.approx(2 * (0.5 ** 10), rel=1e-9)


def test_symmetric_split_gives_p_one():
    a = [1.0, 0.0, 1.0, 0.0]
    b = [0.0, 1.0, 0.0, 1.0]
    result = sign_test(a, b)
    assert result["a_wins"] == 2
    assert result["b_wins"] == 2
    assert result["p_value"] == 1.0


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        sign_test([1.0, 0.0], [1.0])


def test_p_value_is_two_sided_and_symmetric_between_a_and_b():
    a = [1.0, 1.0, 1.0, 0.0, 0.0]
    b = [0.0, 0.0, 0.0, 1.0, 0.0]
    result_ab = sign_test(a, b)
    result_ba = sign_test(b, a)
    assert result_ab["p_value"] == result_ba["p_value"]
    assert result_ab["a_wins"] == result_ba["b_wins"]
