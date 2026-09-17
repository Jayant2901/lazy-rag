import json

import numpy as np
import pytest

from src.retriever import Retriever


class FakeSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        # deterministic embedding: [len(text), 0.0] so nearest-neighbor order is predictable
        return np.array([[float(len(t)), 0.0] for t in texts], dtype="float32")


class FakeIndex:
    def __init__(self, dim):
        self.dim = dim
        self.vectors = None

    def add(self, vectors):
        self.vectors = np.array(vectors, dtype="float32")

    def search(self, query, k):
        dists = np.linalg.norm(self.vectors - query, axis=1)
        order = np.argsort(dists)[:k]
        scores = -dists[order]
        idxs = order.astype("int64")
        if len(idxs) < k:
            pad = k - len(idxs)
            idxs = np.concatenate([idxs, np.full(pad, -1, dtype="int64")])
            scores = np.concatenate([scores, np.zeros(pad, dtype="float32")])
        return scores.reshape(1, -1), idxs.reshape(1, -1)


@pytest.fixture(autouse=True)
def patch_retriever_deps(monkeypatch):
    monkeypatch.setattr("src.retriever.SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr("src.retriever.faiss.IndexFlatIP", FakeIndex)
    monkeypatch.setattr("src.retriever.faiss.write_index", lambda index, path: None)
    monkeypatch.setattr("src.retriever.faiss.read_index", lambda path: None)


def write_corpus(tmp_path, docs, name="corpus.jsonl"):
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    return path


def test_cache_path_deterministic_and_content_sensitive(tmp_path):
    path1 = write_corpus(tmp_path, [{"id": "0", "text": "a"}])
    r = Retriever(str(path1), cache_dir=str(tmp_path))
    p1 = r._cache_path(str(path1), str(tmp_path))
    p1_again = r._cache_path(str(path1), str(tmp_path))
    assert p1 == p1_again

    path2 = write_corpus(tmp_path, [{"id": "0", "text": "b"}], name="corpus2.jsonl")
    p2 = r._cache_path(str(path2), str(tmp_path))
    assert p1 != p2


def test_retrieve_filters_out_negative_one_indices(tmp_path):
    docs = [{"id": "0", "text": "alpha"}, {"id": "1", "text": "beta"}]
    path = write_corpus(tmp_path, docs)
    r = Retriever(str(path), cache_dir=str(tmp_path))

    results = r.retrieve("query text", k=5)  # k > corpus size forces -1 padding

    assert len(results) == len(docs)
    assert all(d in docs for d in results)


def test_recall_at_k_counts_only_items_with_gold_title(tmp_path, monkeypatch):
    docs = [{"id": "0", "title": "A", "text": "a"}, {"id": "1", "title": "B", "text": "b"}]
    path = write_corpus(tmp_path, docs)
    r = Retriever(str(path), cache_dir=str(tmp_path))

    def fake_retrieve(query, k=3):
        return [{"title": "A"}] if query == "hit" else [{"title": "B"}]

    monkeypatch.setattr(r, "retrieve", fake_retrieve)

    qa_set = [
        {"question": "hit", "gold_title": "A"},
        {"question": "miss", "gold_title": "A"},
        {"question": "no gold field"},
    ]
    assert r.recall_at_k(qa_set, k=3) == 0.5


def test_recall_at_k_returns_zero_when_no_gold_titles(tmp_path):
    docs = [{"id": "0", "title": "A", "text": "a"}]
    path = write_corpus(tmp_path, docs)
    r = Retriever(str(path), cache_dir=str(tmp_path))
    assert r.recall_at_k([{"question": "q"}], k=3) == 0.0
