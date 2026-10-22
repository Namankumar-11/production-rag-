"""
Cross-encoder reranking.

Bi-encoders (what VectorRetriever uses) embed the query and each
document *independently*, then compare vectors. That's what makes
them fast enough to search over thousands of chunks -- but it also
means the model never actually looks at the query and document
together, so relevance judgments are coarser.

A cross-encoder takes (query, chunk) pairs *jointly* as input and
outputs a single relevance score. Far more accurate, but too slow to
run over an entire corpus. So the standard production pattern is:

    cheap hybrid retrieval (bm25 + vector) -> top 20 candidates
    -> expensive cross-encoder rerank -> top 5 final chunks

This is exactly the two-stage "retrieve then rerank" architecture
used by real search systems (and is a very common interview question:
"why not just use the cross-encoder for everything?" -- because its
O(n) cost over the full corpus doesn't scale.)
"""
from typing import List, Tuple

from sentence_transformers import CrossEncoder

from app.config import settings
from app.ingestion.chunker import Chunk

_model_cache = {}


def _get_model(model_name: str) -> CrossEncoder:
    if model_name not in _model_cache:
        _model_cache[model_name] = CrossEncoder(model_name)
    return _model_cache[model_name]


class Reranker:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        self.model = _get_model(self.model_name)

    def rerank(self, query: str, chunks: List[Chunk], top_k: int = None) -> List[Tuple[Chunk, float]]:
        top_k = top_k or settings.TOP_K_RERANK
        if not chunks:
            return []

        pairs = [(query, c.text) for c in chunks]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
