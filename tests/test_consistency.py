from src.consistency import consistency_score


def test_unanimous_samples_gives_confidence_one():
    answer, confidence = consistency_score(["Paris", "Paris", "Paris"])
    assert answer == "Paris"
    assert confidence == 1.0


def test_majority_wins_and_confidence_is_fraction():
    answer, confidence = consistency_score(["Paris", "Paris", "Lyon"])
    assert answer == "Paris"
    assert confidence == 2 / 3


def test_all_different_gives_confidence_one_over_n():
    answer, confidence = consistency_score(["Paris", "Lyon", "Nice"])
    assert confidence == 1 / 3


def test_case_and_punctuation_insensitive_agreement():
    answer, confidence = consistency_score(["Paris.", "PARIS", "paris"])
    assert confidence == 1.0


def test_single_sample_gives_confidence_one():
    answer, confidence = consistency_score(["Paris"])
    assert answer == "Paris"
    assert confidence == 1.0
