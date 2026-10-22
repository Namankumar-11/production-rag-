"""
Evaluation runner.

Loads the labeled eval set, runs the hybrid+rerank retriever for every
query, and prints Recall@k / Precision@k / MRR / Hit Rate@k averaged
across the whole set. Run this any time you change chunking,
retrieval weights, or the reranker model, to catch regressions before
they reach production.

Usage:
    python -m evaluation.run_eval
"""
import json
from pathlib import Path

from evaluation.metrics import (
    hit_rate_at_k,
    mean_metric,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
)
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_directory
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.vector_retriever import VectorRetriever

K = 5
DATA_DIR = "data/sample_docs"
EVAL_SET_PATH = Path(__file__).parent / "eval_dataset.json"


def run():
    docs = load_directory(DATA_DIR)
    chunks = chunk_documents(docs)

    bm25 = BM25Retriever()
    bm25.build(chunks)
    vector = VectorRetriever()
    vector.build(chunks)
    hybrid = HybridRetriever(bm25, vector)
    reranker = Reranker()

    eval_set = json.loads(EVAL_SET_PATH.read_text())

    recalls, precisions, rrs, hits = [], [], [], []

    print(f"{'Query':<55} {'Recall@k':<10} {'Prec@k':<10} {'RR':<8} {'Hit@k':<6}")
    print("-" * 95)

    for item in eval_set:
        query = item["query"]
        relevant_doc_ids = set(item["relevant_doc_ids"])

        candidates = hybrid.search(query)
        reranked = reranker.rerank(query, candidates, top_k=K)
        retrieved_ids = [c.chunk_id for c, _ in reranked]

        # a chunk is "relevant" if its parent doc_id is in the labeled relevant set
        relevant_chunk_ids = [c.chunk_id for c in chunks if c.doc_id in relevant_doc_ids]

        r = recall_at_k(retrieved_ids, relevant_chunk_ids, K)
        p = precision_at_k(retrieved_ids, relevant_chunk_ids, K)
        rr = reciprocal_rank(retrieved_ids, relevant_chunk_ids)
        h = hit_rate_at_k(retrieved_ids, relevant_chunk_ids, K)

        recalls.append(r)
        precisions.append(p)
        rrs.append(rr)
        hits.append(h)

        print(f"{query[:53]:<55} {r:<10.2f} {p:<10.2f} {rr:<8.2f} {h:<6.0f}")

    print("-" * 95)
    print(f"{'MEAN':<55} {mean_metric(recalls):<10.2f} {mean_metric(precisions):<10.2f} "
          f"{mean_metric(rrs):<8.2f} {mean_metric(hits):<6.2f}")


if __name__ == "__main__":
    run()
