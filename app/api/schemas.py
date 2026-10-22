from typing import List

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class SourceItem(BaseModel):
    chunk_id: str
    source: str
    text: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceItem]
    grounded: bool
    ungrounded_sentences: List[str]
    num_candidates_before_rerank: int


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
