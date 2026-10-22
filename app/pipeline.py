"""
Pipeline orchestration.

This is the single place that stitches ingestion -> retrieval ->
reranking -> generation -> citation verification into one callable
object. Keeping orchestration separate from each component is what
makes the components independently testable (see tests/) and
independently swappable (e.g. replace FAISS with a hosted vector DB
without touching this file).
"""
from dataclasses import dataclass, field
from typing import List

from app.config import settings
from app.generation.citation import CitationReport, verify_citations
from app.generation.generator import Generator
from app.ingestion.chunker import Chunk, chunk_documents
from app.ingestion.loader import load_directory
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.vector_retriever import VectorRetriever


@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: List[Chunk]
    citation_report: CitationReport
    retrieved_before_rerank: int = 0


class RAGPipeline:
    def __init__(self, llm_provider: str = None):
        self.bm25 = BM25Retriever()
        self.vector = VectorRetriever()
        self.hybrid = HybridRetriever(self.bm25, self.vector)
        self.reranker = Reranker()
        self.generator = Generator(provider=llm_provider)
        self._built = False

    # ---------- Index building ----------

    def build_from_directory(self, dir_path: str) -> None:
        docs = load_directory(dir_path)
        chunks = chunk_documents(docs)
        self.build_from_chunks(chunks)

    def build_from_chunks(self, chunks: List[Chunk]) -> None:
        self.bm25.build(chunks)
        self.vector.build(chunks)
        self._built = True

    def save(self) -> None:
        self.bm25.save(settings.BM25_INDEX_PATH)
        self.vector.save(settings.VECTOR_INDEX_PATH, settings.METADATA_PATH)

    def load(self) -> None:
        self.bm25.load(settings.BM25_INDEX_PATH)
        self.vector.load(settings.VECTOR_INDEX_PATH, settings.METADATA_PATH)
        self._built = True

    # ---------- Query ----------

    def query(self, question: str) -> RAGResponse:
        if not self._built:
            raise RuntimeError("Pipeline index not built. Call build_from_directory() or load() first.")

        candidates = self.hybrid.search(question)
        reranked = self.reranker.rerank(question, candidates)
        top_chunks = [c for c, _score in reranked]

        answer = self.generator.generate(question, top_chunks)
        report = verify_citations(answer, top_chunks)

        return RAGResponse(
            query=question,
            answer=answer,
            sources=top_chunks,
            citation_report=report,
            retrieved_before_rerank=len(candidates),
        )
