"""
Generation layer.

Deliberately provider-agnostic: swapping Groq -> OpenAI -> a local
model should never require touching the pipeline logic. This is the
Strategy pattern applied to LLM calls, which matters a lot in
production because API providers change/deprecate models often and
you don't want that churn touching business logic.

The prompt template forces the model to (a) answer only from the
provided context and (b) tag every claim with a [n] citation marker
referencing the numbered source chunk. That structured citation
format is what `citation.py` verifies downstream.
"""
from typing import List

from app.config import settings
from app.ingestion.chunker import Chunk

SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer the user's question using ONLY the numbered context chunks provided below.
Rules:
- Every factual sentence in your answer MUST end with a citation marker like [1] or [2] referencing the chunk it came from.
- If the answer is not contained in the context, say so explicitly. Do not use outside knowledge.
- Be concise and direct."""


def _build_context_block(chunks: List[Chunk]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[{i}] (source: {chunk.source})\n{chunk.text}")
    return "\n\n".join(lines)


def build_prompt(query: str, chunks: List[Chunk]) -> str:
    context = _build_context_block(chunks)
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer (cite every claim with [n]):"


class Generator:
    def __init__(self, provider: str = None, model: str = None):
        self.provider = provider or settings.LLM_PROVIDER
        self.model = model or settings.LLM_MODEL

    def generate(self, query: str, chunks: List[Chunk]) -> str:
        prompt = build_prompt(query, chunks)

        if self.provider == "mock":
            # deterministic offline mode used by tests/CI, no API key required
            return self._mock_generate(chunks)

        if self.provider == "groq":
            return self._groq_generate(prompt)

        if self.provider == "openai":
            return self._openai_generate(prompt)

        raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")

    def _mock_generate(self, chunks: List[Chunk]) -> str:
        if not chunks:
            return "The context does not contain enough information to answer."
        return f"Based on the provided sources, here is a summary. [1]"

    def _groq_generate(self, prompt: str) -> str:
        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content

    def _openai_generate(self, prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content
