import pytest

from app.ingestion.chunker import Chunk
from app.retrieval.bm25_retriever import BM25Retriever


@pytest.fixture
def sample_chunks():
    return [
        Chunk(chunk_id="a::0", doc_id="a", source="a.txt", text="The cat sat on the mat in the sun.", position=0),
        Chunk(chunk_id="b::0", doc_id="b", source="b.txt", text="Python is a popular programming language.", position=0),
        Chunk(chunk_id="c::0", doc_id="c", source="c.txt", text="Dogs and cats are common household pets.", position=0),
    ]


def test_build_and_search_returns_ranked_results(sample_chunks):
    retriever = BM25Retriever()
    retriever.build(sample_chunks)
    results = retriever.search("programming language", top_k=3)
    assert len(results) == 3
    top_chunk, top_score = results[0]
    assert top_chunk.chunk_id == "b::0"


def test_search_before_build_raises():
    retriever = BM25Retriever()
    with pytest.raises(RuntimeError):
        retriever.search("anything")


def test_top_k_limits_results(sample_chunks):
    retriever = BM25Retriever()
    retriever.build(sample_chunks)
    results = retriever.search("cat", top_k=1)
    assert len(results) == 1


def test_save_and_load_roundtrip(tmp_path, sample_chunks):
    retriever = BM25Retriever()
    retriever.build(sample_chunks)
    path = tmp_path / "bm25.pkl"
    retriever.save(str(path))

    loaded = BM25Retriever()
    loaded.load(str(path))
    results = loaded.search("cat")
    assert len(results) == 3
