import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import EMBED_MODEL


class Retriever:
    def __init__(self, corpus_path: str):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.docs = [json.loads(line) for line in open(corpus_path, encoding="utf-8")]
        embeddings = self.model.encode(
            [d["text"] for d in self.docs], normalize_embeddings=True, show_progress_bar=True
        )
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.array(embeddings, dtype="float32"))

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        query_vec = self.model.encode([query], normalize_embeddings=True)
        scores, idxs = self.index.search(np.array(query_vec, dtype="float32"), k)
        return [self.docs[i] for i in idxs[0] if i != -1]
