"""
Chunking layer.

This is one of the highest-leverage design decisions in a RAG system
and the thing most tutorials get wrong by just splitting on a fixed
character count. Bad chunking silently destroys retrieval quality
because it cuts sentences/ideas in half.

We do two things properly here:
1. Split on sentence boundaries first (never break mid-sentence).
2. Pack sentences into chunks up to CHUNK_SIZE tokens, with
   CHUNK_OVERLAP tokens repeated at the start of the next chunk so a
   fact sitting near a chunk boundary is never orphaned from its
   context in both neighboring chunks.

We approximate "tokens" with whitespace word count. This is a
deliberate simplification -- swapping in a real tokenizer (tiktoken /
the embedding model's tokenizer) is a one-line change in
`_word_count`, and is called out in the README as a known
production upgrade.
"""
import re
from dataclasses import dataclass
from typing import List

from app.config import settings
from app.ingestion.loader import RawDocument


@dataclass
class Chunk:
    chunk_id: str        # f"{doc_id}::{index}"
    doc_id: str
    source: str
    text: str
    position: int         # order within the parent document, used for debugging/traceability


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return _SENTENCE_SPLIT_RE.split(text)


def _word_count(s: str) -> int:
    return len(s.split())


def chunk_document(doc: RawDocument, chunk_size: int = None, overlap: int = None) -> List[Chunk]:
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    sentences = _split_sentences(doc.text)
    chunks: List[Chunk] = []
    current: List[str] = []
    current_len = 0
    position = 0

    for sentence in sentences:
        s_len = _word_count(sentence)

        if current_len + s_len > chunk_size and current:
            chunk_text = " ".join(current)
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}::{position}",
                doc_id=doc.doc_id,
                source=doc.source,
                text=chunk_text,
                position=position,
            ))
            position += 1

            # carry the tail of the previous chunk forward as overlap
            overlap_sentences = []
            overlap_len = 0
            for sent in reversed(current):
                overlap_len += _word_count(sent)
                overlap_sentences.insert(0, sent)
                if overlap_len >= overlap:
                    break
            current = overlap_sentences
            current_len = overlap_len

        current.append(sentence)
        current_len += s_len

    if current:
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}::{position}",
            doc_id=doc.doc_id,
            source=doc.source,
            text=" ".join(current),
            position=position,
        ))

    return chunks


def chunk_documents(docs: List[RawDocument]) -> List[Chunk]:
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
