"""
Central configuration for the RAG pipeline.

Why this file exists:
Production systems never hardcode magic numbers inside logic files.
Every tunable knob (chunk size, model names, retrieval weights) lives
here so you can change behavior without touching pipeline code, and
so the same code can be pointed at different environments (dev/prod)
via environment variables.
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # --- Chunking ---
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 512))          # tokens per chunk
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 64))     # overlap to preserve context across boundaries

    # --- Embedding model (dense retrieval) ---
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # --- Cross-encoder reranker ---
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # --- Hybrid retrieval weights (must sum to 1.0) ---
    BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", 0.4))
    VECTOR_WEIGHT: float = float(os.getenv("VECTOR_WEIGHT", 0.6))

    # --- Retrieval depth ---
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", 20))   # candidates pulled before reranking
    TOP_K_RERANK: int = int(os.getenv("TOP_K_RERANK", 5))          # final chunks passed to the LLM

    # --- Generation ---
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")          # groq | openai | mock
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # --- Storage paths ---
    VECTOR_INDEX_PATH: str = os.getenv("VECTOR_INDEX_PATH", "data/index/faiss.index")
    METADATA_PATH: str = os.getenv("METADATA_PATH", "data/index/metadata.jsonl")
    BM25_INDEX_PATH: str = os.getenv("BM25_INDEX_PATH", "data/index/bm25.pkl")

    # --- Citation enforcement ---
    MIN_CITATION_OVERLAP: float = float(os.getenv("MIN_CITATION_OVERLAP", 0.15))


settings = Settings()
