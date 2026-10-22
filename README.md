# Production RAG

A hybrid-retrieval RAG system built the way a real search/AI team would build it:
BM25 + dense vector retrieval fused with Reciprocal Rank Fusion, cross-encoder
reranking, programmatic citation enforcement to catch hallucination, a labeled
retrieval evaluation pipeline, and a tested FastAPI service.

## Architecture

```
 documents
    │
    ▼
 ┌─────────────┐
 │  Chunker    │  sentence-aware, overlapping chunks
 └─────────────┘
    │
    ▼
 ┌──────────────────────┬───────────────────────┐
 │   BM25 (lexical)     │   FAISS (semantic)     │   ← built in parallel
 └──────────────────────┴───────────────────────┘
                  │
                  ▼
        Reciprocal Rank Fusion (top 20)
                  │
                  ▼
        Cross-Encoder Reranker (top 5)
                  │
                  ▼
              LLM Generation
             (forced citations)
                  │
                  ▼
       Citation Verification
      (lexical grounding check)
                  │
                  ▼
              Final Answer
```

## Why each piece exists

| Component | Problem it solves |
|---|---|
| Sentence-aware chunking with overlap | Naive character-count splitting cuts facts in half and orphans context at chunk boundaries |
| BM25 | Dense embeddings are weak on exact terms (codes, acronyms, numbers) |
| Vector search | BM25 is weak on semantic/synonym matches |
| Reciprocal Rank Fusion | BM25 and cosine scores live on incomparable scales — can't just add them |
| Cross-encoder reranking | Bi-encoders can't jointly attend over query+doc; cross-encoders are far more accurate but too slow to run over the whole corpus, so it's a two-stage funnel |
| Citation enforcement | LLMs cite confidently even when wrong — this catches it programmatically instead of trusting the model |
| Evaluation pipeline | You can't improve retrieval quality you never measured |
| CI (pytest + GitHub Actions) | Every pipeline change (chunk size, weights, model swap) is checked against regressions automatically |

## Project layout

```
app/
  config.py                 all tunable settings, env-var driven
  ingestion/
    loader.py                reads raw documents
    chunker.py                sentence-aware chunking with overlap
  retrieval/
    bm25_retriever.py         sparse lexical search
    vector_retriever.py       dense embedding search (FAISS)
    hybrid_retriever.py       RRF fusion of both
    reranker.py                cross-encoder reranking
  generation/
    generator.py               provider-agnostic LLM call (Groq/OpenAI/mock)
    citation.py                 grounding verification
  pipeline.py                  orchestrates the full flow
  api/
    main.py                    FastAPI app
    schemas.py                  request/response models
evaluation/
  metrics.py                    recall@k, precision@k, MRR, hit rate@k
  eval_dataset.json              labeled query -> relevant doc set
  run_eval.py                     evaluation runner
scripts/
  build_index.py                 builds and persists BM25 + FAISS indexes
tests/                            16 unit tests, fully offline (mocked models)
.github/workflows/ci.yml          runs tests + eval on every push
Dockerfile
```

## Running it locally

```bash
pip install -r requirements.txt

# 1. Build the indexes from data/sample_docs/
python -m scripts.build_index

# 2. Run the evaluation suite against the real models
python -m evaluation.run_eval

# 3. Run the test suite (fast, offline, mocked models)
pytest tests/ -v

# 4. Start the API
export GROQ_API_KEY=your_key_here   # or set LLM_PROVIDER=mock to test without a key
uvicorn app.api.main:app --reload

# 5. Query it
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Why is chunking important in RAG?"}'
```

## Known limitations / natural next upgrades

Being upfront about these is a good interview signal — it shows you understand
the tradeoffs you made, not just that you copy-pasted a tutorial.

- **Token counting** uses whitespace word count as an approximation, not the
  actual embedding model's tokenizer. Swap in `tiktoken` or the model's own
  tokenizer for exact context-window budgeting.
- **Citation checking is lexical (word overlap), not semantic.** It catches
  "cited the wrong chunk" and "made up a claim with no support" but won't
  catch a subtly wrong paraphrase. An LLM-as-judge second pass is the natural
  upgrade for that, at the cost of latency and money.
- **FAISS IndexFlatIP is exact search**, fine at portfolio scale (thousands of
  chunks). A real production corpus (millions of chunks) would need an ANN
  index (HNSW/IVF) and a discussion of the recall/speed tradeoff.
- **No caching layer.** Repeated identical queries re-run the full pipeline;
  a production system would cache embeddings and even full responses.
- **Single-node.** No mention of horizontal scaling, load balancing, or a
  managed vector DB (Pinecone/Weaviate/Qdrant) — worth knowing how you'd
  evolve toward that if asked.

## Test coverage

16 tests, all offline (embedding/reranker models are mocked via
`tests/conftest.py` with deterministic fakes) so CI never depends on
network access or GPU:
- chunking correctness (boundaries, overlap, ids, empty input)
- BM25 build/search/persistence
- hybrid RRF fusion ranking behavior
- citation grounding (including the sentence/citation-boundary edge case)
