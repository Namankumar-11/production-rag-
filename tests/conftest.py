"""
Shared test fixtures.

Unit tests should never depend on network access or multi-hundred-MB
model downloads -- that's slow, flaky, and couples test correctness to
an external service's uptime. We fake the embedding model and
cross-encoder with tiny deterministic stand-ins that still produce
meaningful relative rankings (via hashed bag-of-words), so retrieval
*logic* (RRF fusion, reranking order) is exercised properly without
ever calling huggingface.co.

Real semantic quality is instead validated by evaluation/run_eval.py
against the real models -- that's an integration/eval concern, not a
unit test concern. This separation (fast deterministic unit tests vs.
slower model-backed evaluation) mirrors how production ML teams
structure their test suites.
"""
import numpy as np
import pytest


class FakeEmbedder:
    """Deterministic hashed bag-of-words 'embedding' -- captures lexical
    overlap well enough to exercise retrieval logic without a real model."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        vectors = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for word in text.lower().split():
                idx = hash(word) % self.dim
                vectors[i, idx] += 1.0
        return vectors


class FakeCrossEncoder:
    """Deterministic relevance score: count of shared words between
    query and candidate. Enough to validate reranking order in tests."""

    def predict(self, pairs):
        scores = []
        for query, doc in pairs:
            q_words = set(query.lower().split())
            d_words = set(doc.lower().split())
            scores.append(float(len(q_words & d_words)))
        return scores


@pytest.fixture(autouse=True)
def _mock_ml_models(monkeypatch):
    """Auto-applied to every test: replaces real model loaders with fakes."""
    import app.retrieval.vector_retriever as vr
    import app.retrieval.reranker as rr

    monkeypatch.setattr(vr, "_get_model", lambda name: FakeEmbedder())
    monkeypatch.setattr(rr, "_get_model", lambda name: FakeCrossEncoder())
