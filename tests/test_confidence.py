from src.confidence import parse_confidence_response


def test_well_formed_response():
    raw = "Answer: Paris\nConfidence: 0.9"
    answer, confidence = parse_confidence_response(raw)
    assert answer == "Paris"
    assert confidence == 0.9


def test_integer_formatted_confidence():
    raw = "Answer: Paris\nConfidence: 1"
    answer, confidence = parse_confidence_response(raw)
    assert confidence == 1.0


def test_missing_confidence_line_defaults_to_zero():
    raw = "Answer: Paris"
    answer, confidence = parse_confidence_response(raw)
    assert answer == "Paris"
    assert confidence == 0.0


def test_missing_answer_line_falls_back_to_raw_text():
    raw = "I'm not sure how to format this."
    answer, confidence = parse_confidence_response(raw)
    assert answer == raw
    assert confidence == 0.0


def test_extra_whitespace_and_multiline_answer():
    raw = "Answer:   Paris, the capital of France   \nConfidence: 0.85"
    answer, confidence = parse_confidence_response(raw)
    assert answer == "Paris, the capital of France"
    assert confidence == 0.85


def test_confidence_with_trailing_text():
    raw = "Answer: Paris\nConfidence: 0.75 (fairly sure)"
    answer, confidence = parse_confidence_response(raw)
    assert confidence == 0.75
