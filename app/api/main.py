"""
FastAPI serving layer.

Loads the pre-built index once at startup (not per-request -- loading
a FAISS index and embedding model on every request would be brutally
slow) and exposes a single /query endpoint. Also exposes /health for
load balancers / uptime checks, which is standard practice for any
service you'd actually deploy.
"""
import logging

from fastapi import FastAPI, HTTPException

from app.api.schemas import HealthResponse, QueryRequest, QueryResponse, SourceItem
from app.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-api")

app = FastAPI(
    title="Production RAG API",
    description="Hybrid retrieval (BM25 + vector) with cross-encoder reranking and citation enforcement.",
    version="1.0.0",
)

pipeline = RAGPipeline()


@app.on_event("startup")
def startup_event():
    try:
        pipeline.load()
        logger.info("RAG index loaded successfully.")
    except FileNotFoundError:
        logger.warning("No prebuilt index found. Call the ingestion script before querying.")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", index_loaded=pipeline._built)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not pipeline._built:
        raise HTTPException(status_code=503, detail="Index not loaded. Run the ingestion script first.")

    try:
        result = pipeline.query(request.question)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return QueryResponse(
        query=result.query,
        answer=result.answer,
        sources=[
            SourceItem(chunk_id=c.chunk_id, source=c.source, text=c.text)
            for c in result.sources
        ],
        grounded=result.citation_report.all_grounded,
        ungrounded_sentences=result.citation_report.ungrounded_sentences,
        num_candidates_before_rerank=result.retrieved_before_rerank,
    )
