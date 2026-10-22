"""
Run this once (or whenever the source documents change) to build and
persist the BM25 and FAISS indexes to disk. The API loads these
prebuilt indexes at startup rather than rebuilding them on every boot.

Usage:
    python -m scripts.build_index
"""
from app.pipeline import RAGPipeline

DATA_DIR = "data/sample_docs"


def main():
    pipeline = RAGPipeline()
    print(f"Loading and chunking documents from {DATA_DIR} ...")
    pipeline.build_from_directory(DATA_DIR)
    print("Saving indexes to disk ...")
    pipeline.save()
    print("Done. Indexes saved to data/index/")


if __name__ == "__main__":
    main()
