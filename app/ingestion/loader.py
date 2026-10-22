"""
Document loading layer.

Real production systems ingest messy heterogeneous sources (PDFs, HTML,
Notion exports, Slack threads...). We keep the loader interface narrow
and format-agnostic on purpose: every loader returns the same
`RawDocument` shape, so everything downstream (chunker, indexer) never
needs to know or care what the original file format was.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class RawDocument:
    doc_id: str          # stable unique id, derived from filename
    source: str           # file path or URL, used for citations
    text: str


def load_text_file(path: str) -> RawDocument:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    return RawDocument(doc_id=p.stem, source=str(p), text=text)


def load_directory(dir_path: str, extensions=(".txt", ".md")) -> List[RawDocument]:
    """Load every matching file in a directory. This is the entry point
    the ingestion pipeline calls to bulk-load a knowledge base."""
    p = Path(dir_path)
    docs = []
    for ext in extensions:
        for file in sorted(p.glob(f"*{ext}")):
            docs.append(load_text_file(str(file)))
    return docs
