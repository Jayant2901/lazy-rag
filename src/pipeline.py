from dataclasses import dataclass
from src.llm import generate
from src.retriever import Retriever
from src.confidence import answer_with_confidence

RAG_PROMPT = """Answer the question using the context below. If the context doesn't
contain the answer, say so rather than guessing.

Context:
{context}

Question: {question}
Answer:"""


@dataclass
class PipelineResult:
    answer: str
    retrieved: bool
    num_retrieved: int = 0


def answer_with_context(question: str, retriever: Retriever, k: int = 3) -> PipelineResult:
    docs = retriever.retrieve(question, k=k)
    context = "\n\n".join(d["text"] for d in docs)
    answer = generate(RAG_PROMPT.format(context=context, question=question))
    return PipelineResult(answer=answer, retrieved=True, num_retrieved=len(docs))


def no_rag(question: str, retriever: Retriever | None = None) -> PipelineResult:
    answer = generate(question)
    return PipelineResult(answer=answer, retrieved=False)


def always_rag(question: str, retriever: Retriever, k: int = 3) -> PipelineResult:
    return answer_with_context(question, retriever, k=k)


def lazy_rag(question: str, retriever: Retriever, threshold: float = 0.6, k: int = 3) -> PipelineResult:
    draft_answer, confidence = answer_with_confidence(question)
    if confidence >= threshold:
        return PipelineResult(answer=draft_answer, retrieved=False)
    return answer_with_context(question, retriever, k=k)
