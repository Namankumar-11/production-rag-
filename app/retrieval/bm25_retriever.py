"""
BM25 (sparse / lexical) retriever.

Why BM25 at all, when we have embeddings?
Dense vector search is great at *semantic* similarity but is
surprisingly bad at exact-match signals: product codes, error
messages, acronyms, numbers, rare proper nouns. BM25 is a statistical
term-frequency method (a smarter version of TF-IDF) and nails exactly
those cases. Combining both (hybrid retrieval) covers each other's
blind spots -- that's the entire thesis of this project.
"""
import pickle
from pathlib import Path
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from app.ingestion.chunker import Chunk


class BM25Retriever:
    def __init__(self):
        self._bm25: BM25Okapi = None
        self._chunks: List[Chunk] = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Lowercasing + simple whitespace split. Good enough for BM25;
        # swap for a proper tokenizer if you need stemming/stopwords.
        return text.lower().split()

    def build(self, chunks: List[Chunk]) -> None:
        self._chunks = chunks
        tokenized_corpus = [self._tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Chunk, float]]:
        if self._bm25 is None:
            raise RuntimeError("BM25Retriever.build() must be called before search()")
        scores = self._bm25.get_scores(self._tokenize(query))
        ranked = sorted(zip(self._chunks, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "chunks": self._chunks}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunks = data["chunks"]
