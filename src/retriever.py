import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import EMBED_MODEL


class Retriever:
    def __init__(self, corpus_path: str, cache_dir: str = "data"):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.docs = [json.loads(line) for line in open(corpus_path, encoding="utf-8")]

        cache_path = self._cache_path(corpus_path, cache_dir)
        if cache_path.exists():
            self.index = faiss.read_index(str(cache_path))
        else:
            embeddings = self.model.encode(
                [d["text"] for d in self.docs], normalize_embeddings=True, show_progress_bar=True
            )
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(np.array(embeddings, dtype="float32"))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(cache_path))

    def _cache_path(self, corpus_path: str, cache_dir: str) -> Path:
        content_hash = hashlib.sha256(
            Path(corpus_path).read_bytes() + EMBED_MODEL.encode()
        ).hexdigest()[:16]
        return Path(cache_dir) / f"{Path(corpus_path).stem}.{content_hash}.index"

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        query_vec = self.model.encode([query], normalize_embeddings=True)
        scores, idxs = self.index.search(np.array(query_vec, dtype="float32"), k)
        return [self.docs[i] for i in idxs[0] if i != -1]

    def recall_at_k(self, qa_set: list[dict], k: int = 3) -> float:
        """Fraction of questions where the gold document appears in the top-k retrieved set.

        Measures retrieval quality in isolation from generation - a question can fail
        end-to-end either because retrieval missed the right document or because the
        model mishandled a correctly retrieved one. Only questions carrying a
        `gold_title` field (set by eval/prepare_popqa.py) are counted.
        """
        hits, total = 0, 0
        for item in qa_set:
            gold_title = item.get("gold_title")
            if not gold_title:
                continue
            total += 1
            retrieved = self.retrieve(item["question"], k=k)
            if any(d.get("title") == gold_title for d in retrieved):
                hits += 1
        return hits / total if total else 0.0
