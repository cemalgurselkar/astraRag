# AstraRAG Progress

## Completed

- [x] Project initialized with Python 3.11 + `uv`
- [x] Runtime dependencies configured
- [x] Development dependencies configured
- [x] Core document schemas
  - [x] `Page`
  - [x] `Document`
  - [x] `Chunk`
  - [x] `RetrievalResult`
- [x] PDF parsing with PyMuPDF
- [x] Deterministic document IDs with SHA-256
- [x] Fixed-size chunking
- [x] Configurable chunk size
- [x] Configurable chunk overlap
- [x] Page metadata preservation
- [x] SentenceTransformer embedding encoder
- [x] Local embedding model loading
- [x] Normalized dense embeddings
- [x] Qdrant dense vector index
- [x] Persistent local Qdrant storage
- [x] Cosine similarity search
- [x] Deterministic Qdrant point IDs with UUID5
- [x] Idempotent document indexing
- [x] Dense retriever
- [x] Query embedding
- [x] Top-K dense retrieval
- [x] End-to-end PDF → Retrieve pipeline
- [x] Manual retrieval sanity check
- [x] Duplicate indexing bug fixed
- [x] Ruff checks passing
- [x] mypy checks passing
- [x] pytest checks passing

## Current

- [ ] Dense retrieval evaluation foundation
  - [ ] Create ground-truth evaluation dataset
  - [ ] Define query categories
  - [ ] Implement Recall@K
  - [ ] Implement MRR
  - [ ] Implement nDCG
  - [ ] Build benchmark runner
  - [ ] Record dense baseline results

## Remaining

### Retrieval Baselines

- [ ] BM25 sparse retrieval
- [ ] Dense vs BM25 benchmark
- [ ] Hybrid retrieval
- [ ] RRF fusion
- [ ] Dense vs BM25 vs Hybrid benchmark

### Ranking & Advanced Retrieval

- [ ] Cross-encoder reranking
- [ ] Reranker candidate-depth experiments
- [ ] Parent-child retrieval
- [ ] Multi-query retrieval
- [ ] Compare advanced retrieval strategies

### Retrieval Optimization

- [ ] Chunk-size experiments
- [ ] Chunk-overlap experiments
- [ ] Embedding-model experiments
- [ ] Top-K experiments
- [ ] Hybrid-weight experiments
- [ ] RRF parameter experiments
- [ ] Indexing throughput measurements
- [ ] Retrieval latency measurements
- [ ] Memory measurements

### Context Engineering

- [ ] Context deduplication
- [ ] Relevance filtering
- [ ] Diversity handling
- [ ] Context ordering
- [ ] Context packing
- [ ] Token-budget handling
- [ ] Context pollution experiments
- [ ] 2K / 4K / 8K / 16K context-budget experiments

### Failure-Driven Advanced Retrieval

- [ ] Retrieval failure analysis
- [ ] Semantic chunking experiment if justified
- [ ] Query rewriting experiment if justified
- [ ] HyDE experiment if justified
- [ ] Query decomposition experiment if justified
- [ ] GraphRAG experiment if justified
- [ ] Hierarchical retrieval experiment if justified
- [ ] Context compression experiment if justified

### Adaptive Routing

- [ ] Query profiler
- [ ] Routing features
- [ ] Static routing baseline
- [ ] Rule-based router
- [ ] LLM router
- [ ] Trained router
- [ ] Router benchmark
- [ ] Strategy distribution metrics
- [ ] Router accuracy metrics
- [ ] Escalation metrics
- [ ] Quality / latency / cost trade-off analysis

### Corrective / Agentic Retrieval

- [ ] Evidence grading
- [ ] Retrieval sufficiency detection
- [ ] Strategy escalation
- [ ] Corrective retrieval loop
- [ ] LangGraph integration if justified

### Runtime Optimization

- [ ] Caching
- [ ] Batching
- [ ] Concurrency
- [ ] Async I/O
- [ ] Model routing
- [ ] Fallback handling
- [ ] Runtime budgets

### Generation

- [ ] Generation interface
- [ ] Gemini integration
- [ ] Provider abstraction
- [ ] Grounded answer generation
- [ ] Citation handling
- [ ] Generation evaluation
- [ ] Correctness metrics
- [ ] Groundedness metrics
- [ ] Citation correctness metrics
- [ ] Context precision / recall metrics

### Observability

- [ ] Per-stage tracing
- [ ] Retrieval strategy logging
- [ ] Retrieval score logging
- [ ] Candidate-count logging
- [ ] Context-token logging
- [ ] Escalation-reason logging
- [ ] Model / prompt version logging
- [ ] Token usage logging
- [ ] Cost logging
- [ ] Error logging
- [ ] Experiment configuration logging

### API & Deployment

- [ ] FastAPI service
- [ ] API schemas
- [ ] Runtime reliability controls
- [ ] Docker setup
- [ ] Integration tests
- [ ] End-to-end tests

### Final Evaluation

- [ ] Reproducible public benchmark
- [ ] Final retrieval benchmark
- [ ] Final generation benchmark
- [ ] Latency benchmark
- [ ] Throughput benchmark
- [ ] Resource benchmark
- [ ] Cost benchmark
- [ ] Ablation studies
- [ ] Pareto analysis
- [ ] Final experiment analysis
- [ ] Final README benchmark tables
- [ ] Final architecture documentation