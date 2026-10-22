"""
Dense vector retriever.

Embeds every chunk with a sentence-transformer model and indexes the
vectors in FAISS for fast approximate nearest-neighbor search. This
retriever handles the *semantic* side of hybrid retrieval -- it finds
chunks that mean the same thing as the query even if they don't share
exact words ("car" vs "automobile").

We use IndexFlatIP (exact inner-product search over normalized
vectors == cosine similarity). For a portfolio-scale corpus (thousands
of chunks) exact search is fast enough and far simpler to reason
about than an ANN index like HNSW -- mentioning that tradeoff
explicitly is a good interview talking point.
"""
import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.ingestion.chunker import Chunk

_model_cache = {}


def _get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


class VectorRetriever:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model = _get_model(self.model_name)
        self.index: faiss.Index = None
        self._chunks: List[Chunk] = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # normalize so inner product == cosine similarity
        faiss.normalize_L2(vectors)
        return vectors.astype("float32")

    def build(self, chunks: List[Chunk]) -> None:
        self._chunks = chunks
        vectors = self._embed([c.text for c in chunks])
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Chunk, float]]:
        if self.index is None:
            raise RuntimeError("VectorRetriever.build() must be called before search()")
        query_vec = self._embed([query])
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self._chunks[idx], float(score)))
        return results

    def save(self, index_path: str, metadata_path: str) -> None:
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            for c in self._chunks:
                f.write(json.dumps(c.__dict__) + "\n")

    def load(self, index_path: str, metadata_path: str) -> None:
        self.index = faiss.read_index(index_path)
        self._chunks = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self._chunks.append(Chunk(**data))
