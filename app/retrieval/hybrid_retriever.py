"""
Hybrid retrieval: fuses BM25 (sparse) and vector (dense) results.

The tricky part of hybrid retrieval isn't running two searches --
it's combining their scores fairly. BM25 scores are unbounded
term-frequency scores; cosine similarity is bounded [-1, 1]. You
cannot just add them together, the scales aren't comparable.

We use Reciprocal Rank Fusion (RRF), the standard, battle-tested
technique used by Elasticsearch, Weaviate, and most production hybrid
search systems. Instead of combining raw scores, RRF combines *ranks*:

    RRF_score(doc) = sum over each ranker of  1 / (k + rank_in_that_ranker)

Because it only cares about rank position (not raw score magnitude),
it sidesteps the scale-mismatch problem entirely and is remarkably
robust in practice. k=60 is the standard constant from the original
paper (Cormack et al., 2009) and dampens the influence of very
top-ranked outliers.
"""
from typing import List

from app.config import settings
from app.ingestion.chunker import Chunk
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.vector_retriever import VectorRetriever

RRF_K = 60


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, vector: VectorRetriever):
        self.bm25 = bm25
        self.vector = vector

    def search(self, query: str, top_k: int = None) -> List[Chunk]:
        top_k = top_k or settings.TOP_K_RETRIEVAL

        bm25_results = self.bm25.search(query, top_k=top_k)
        vector_results = self.vector.search(query, top_k=top_k)

        rrf_scores = {}
        chunk_lookup = {}

        for rank, (chunk, _score) in enumerate(bm25_results):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) \
                + settings.BM25_WEIGHT * (1.0 / (RRF_K + rank + 1))
            chunk_lookup[chunk.chunk_id] = chunk

        for rank, (chunk, _score) in enumerate(vector_results):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) \
                + settings.VECTOR_WEIGHT * (1.0 / (RRF_K + rank + 1))
            chunk_lookup[chunk.chunk_id] = chunk

        ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        return [chunk_lookup[cid] for cid in ranked_ids[:top_k]]
