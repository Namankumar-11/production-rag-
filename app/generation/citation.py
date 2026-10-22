"""
Citation enforcement.

This is the piece that separates a "toy RAG demo" from something you'd
actually trust in production. LLMs will confidently cite [1] next to
a sentence that [1] doesn't actually support -- prompting alone does
not guarantee grounding. So we verify it programmatically:

1. Parse the answer into (sentence, cited_chunk_numbers) pairs.
2. For every citation, compute lexical word-overlap between the
   sentence and the cited chunk's text.
3. If overlap falls below MIN_CITATION_OVERLAP, flag it as an
   ungrounded / hallucinated citation.

This is intentionally a lightweight lexical check rather than a second
LLM call ("LLM-as-judge") -- it's fast, free, deterministic, and
catches the most common failure mode (citing the wrong chunk number,
or citing nothing at all) without adding latency or cost to every
request. The README documents LLM-as-judge as the natural upgrade path
for catching subtler semantic hallucinations.
"""
import re
from dataclasses import dataclass
from typing import List

from app.config import settings
from app.ingestion.chunker import Chunk

_CITATION_RE = re.compile(r"\[(\d+)\]")

# Matches a sentence body up through its terminal punctuation AND any
# citation markers immediately trailing it (e.g. "Paris. [1]"), so a
# naive split doesn't sever "[1]" from the sentence it's citing.
_SENTENCE_WITH_CITATIONS_RE = re.compile(r"(.*?[.!?](?:\s*\[\d+\])*)(?:\s+|$)")

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "and", "or", "for", "with", "this", "that", "it", "as", "by", "be",
}


@dataclass
class CitationCheck:
    sentence: str
    cited_indices: List[int]
    grounded: bool
    max_overlap: float


@dataclass
class CitationReport:
    checks: List[CitationCheck]
    all_grounded: bool
    ungrounded_sentences: List[str]


def _words(text: str) -> set:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}


def _overlap_ratio(sentence: str, chunk_text: str) -> float:
    sentence_words = _words(sentence)
    if not sentence_words:
        return 1.0  # nothing to verify (e.g. pure punctuation) -> don't penalize
    chunk_words = _words(chunk_text)
    if not chunk_words:
        return 0.0
    overlap = sentence_words & chunk_words
    return len(overlap) / len(sentence_words)


def _split_sentences_keeping_citations(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = []
    pos = 0
    for m in _SENTENCE_WITH_CITATIONS_RE.finditer(text):
        s = m.group(1).strip()
        if s:
            sentences.append(s)
        pos = m.end()
    if pos < len(text):
        remainder = text[pos:].strip()
        if remainder:
            sentences.append(remainder)
    return sentences


def verify_citations(answer: str, cited_chunks: List[Chunk]) -> CitationReport:
    """
    `cited_chunks` must be in the same order as the [n] markers used
    in the prompt (i.e. cited_chunks[0] == "[1]").
    """
    sentences = _split_sentences_keeping_citations(answer.strip())
    checks: List[CitationCheck] = []

    for sentence in sentences:
        if not sentence.strip():
            continue
        cited_numbers = [int(n) for n in _CITATION_RE.findall(sentence)]
        clean_sentence = _CITATION_RE.sub("", sentence).strip()

        if not cited_numbers:
            checks.append(CitationCheck(clean_sentence, [], grounded=False, max_overlap=0.0))
            continue

        best_overlap = 0.0
        for n in cited_numbers:
            idx = n - 1
            if 0 <= idx < len(cited_chunks):
                overlap = _overlap_ratio(clean_sentence, cited_chunks[idx].text)
                best_overlap = max(best_overlap, overlap)

        grounded = best_overlap >= settings.MIN_CITATION_OVERLAP
        checks.append(CitationCheck(clean_sentence, cited_numbers, grounded, best_overlap))

    ungrounded = [c.sentence for c in checks if not c.grounded]
    return CitationReport(
        checks=checks,
        all_grounded=len(ungrounded) == 0,
        ungrounded_sentences=ungrounded,
    )
