"""
Retrieval evaluation metrics.

You cannot improve what you don't measure. Most "RAG projects" skip
this entirely and just eyeball a few answers. A production team
instead keeps a labeled eval set (query -> which chunk_ids are
actually relevant) and tracks these standard IR metrics every time
the pipeline changes, so a regression (e.g. tweaking chunk size)
shows up as a number dropping, not a vibe.

- Recall@k: of all relevant chunks, what fraction did we retrieve in
  the top k? Measures whether the answer material is even present.
- Precision@k: of the top k retrieved, what fraction were relevant?
  Measures how much noise the LLM has to wade through.
- MRR (Mean Reciprocal Rank): rewards putting the first relevant
  result as early as possible -- 1/rank of the first hit, averaged
  across queries.
- Hit Rate@k: did AT LEAST ONE relevant chunk appear in the top k?
  A coarser, easy-to-communicate sanity metric.
"""
from typing import List


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(retrieved_top_k & relevant) / len(relevant)


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if k == 0:
        return 0.0
    retrieved_top_k = retrieved_ids[:k]
    if not retrieved_top_k:
        return 0.0
    relevant = set(relevant_ids)
    hits = sum(1 for cid in retrieved_top_k if cid in relevant)
    return hits / len(retrieved_top_k)


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    relevant = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    relevant = set(relevant_ids)
    retrieved_top_k = set(retrieved_ids[:k])
    return 1.0 if retrieved_top_k & relevant else 0.0


def mean_metric(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0
