from unittest.mock import patch

from src.pipeline import PipelineResult, lazy_rag


def test_lazy_rag_skips_retrieval_when_confidence_above_threshold():
    with patch("src.pipeline.answer_with_confidence", return_value=("draft", 0.9)), \
         patch("src.pipeline.answer_with_context") as mock_ctx:
        result = lazy_rag("q", retriever=None, threshold=0.6)

    mock_ctx.assert_not_called()
    assert result == PipelineResult(answer="draft", retrieved=False)


def test_lazy_rag_retrieves_when_confidence_below_threshold():
    with patch("src.pipeline.answer_with_confidence", return_value=("draft", 0.59)), \
         patch("src.pipeline.answer_with_context", return_value=PipelineResult("rag answer", True)) as mock_ctx:
        result = lazy_rag("q", retriever="R", threshold=0.6, k=5)

    mock_ctx.assert_called_once_with("q", "R", k=5)
    assert result == PipelineResult(answer="rag answer", retrieved=True)


def test_lazy_rag_boundary_confidence_equal_to_threshold_skips_retrieval():
    """confidence >= threshold in src/pipeline.py, so an exact match counts as confident."""
    with patch("src.pipeline.answer_with_confidence", return_value=("draft", 0.6)), \
         patch("src.pipeline.answer_with_context") as mock_ctx:
        result = lazy_rag("q", retriever=None, threshold=0.6)

    mock_ctx.assert_not_called()
    assert result.retrieved is False
