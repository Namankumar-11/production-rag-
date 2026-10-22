import pytest

from app.ingestion.chunker import Chunk
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_retriever import VectorRetriever


@pytest.fixture
def sample_chunks():
    return [
        Chunk(chunk_id="a::0", doc_id="a", source="a.txt",
              text="BM25 is a lexical keyword search ranking function.", position=0),
        Chunk(chunk_id="b::0", doc_id="b", source="b.txt",
              text="Dense embeddings capture semantic meaning for search.", position=0),
        Chunk(chunk_id="c::0", doc_id="c", source="c.txt",
              text="Bananas are a tropical fruit rich in potassium.", position=0),
    ]


def test_hybrid_search_returns_relevant_chunk_for_lexical_query(sample_chunks):
    bm25 = BM25Retriever()
    bm25.build(sample_chunks)
    vector = VectorRetriever()
    vector.build(sample_chunks)
    hybrid = HybridRetriever(bm25, vector)

    results = hybrid.search("keyword ranking function", top_k=3)
    result_ids = [c.chunk_id for c in results]
    assert "a::0" in result_ids


def test_hybrid_search_excludes_irrelevant_top_result(sample_chunks):
    bm25 = BM25Retriever()
    bm25.build(sample_chunks)
    vector = VectorRetriever()
    vector.build(sample_chunks)
    hybrid = HybridRetriever(bm25, vector)

    results = hybrid.search("semantic embedding search", top_k=1)
    assert results[0].chunk_id != "c::0"
