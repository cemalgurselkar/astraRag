# AstraRAG

AstraRAG is an experiment-driven RAG engine for benchmarking retrieval strategies, context construction, and adaptive query routing.

The project focuses on a simple question:

> Where does a RAG pipeline lose retrieval quality, latency, or efficiency, and which techniques actually improve those trade-offs?

Instead of adding techniques by default, AstraRAG follows a **measure → experiment → keep/reject** workflow.

## Architecture

```text
Documents
   │
   ▼
PDF Parser → Cleaner → Chunker
   │
   ▼
llama.cpp Embedding Server
   │
   ├───────────────┐
   ▼               ▼
Qdrant          BM25 Index
   │               │
   └───────┬───────┘
           │
Query → Profiler → Router
           │
        ┌──┴────┐
        ▼       ▼
      BM25    Hybrid
        │       │
        └───┬───┘
           ▼
     Context Engine
           │
           ▼
     Gemini Generator
           │
           ▼
   Answer + Sources
```

The outer pipeline is deterministic. The LLM is used for grounded answer generation, while retrieval and routing remain measurable system components.

## Features

- PDF parsing and text cleaning
- Fixed-size and parent-child chunking
- all-MiniLM-L6-v2 embeddings served by llama.cpp without Torch
- Versioned Qdrant collections with atomic alias replacement
- Explicit HNSW ANN indexing and search parameters
- Dense and BM25 retrieval
- Hybrid retrieval with Reciprocal Rank Fusion
- Parent-child retrieval
- Rule-based adaptive routing
- Context deduplication and budgeting
- Gemini-based grounded generation
- Retrieval and context benchmarking
- Python, CLI, REST API, and Docker interfaces

Embedding inference runs locally through llama.cpp. Only generation requires the external Gemini API.

## Experiments

AstraRAG uses a 40-query development benchmark over five academic papers.

The historical fixed-size baseline was **1800 characters / 200 overlap**. Production indexing now uses **254 embedding tokens / 32 overlap** so all-MiniLM-L6-v2 never silently truncates most of a chunk. Retrieval scores must be re-baselined for this production configuration.

### Retrieval Benchmark

| Strategy | Hit@5 | Hit@10 | MRR | nDCG@10 | Mean Latency |
|---|---:|---:|---:|---:|---:|
| Dense | 0.575 | 0.725 | 0.464 | 0.510 | 8.02 ms |
| BM25 | 0.750 | 0.825 | 0.621 | 0.663 | 0.73 ms |
| Hybrid RRF | **0.800** | **0.850** | 0.570 | 0.631 | 9.75 ms |
| Hybrid + Cross-Encoder | 0.675 | 0.800 | 0.597 | 0.641 | 1197.42 ms |
| Parent-Child | 0.675 | 0.750 | 0.463 | 0.531 | 9.63 ms |

The experiments show that more complex retrieval is not automatically better. BM25 remains an extremely strong low-latency baseline, while Hybrid RRF provides the highest recall on the current benchmark.

Cross-encoder reranking improved ranking quality under an earlier chunking configuration but became significantly more expensive and less effective after the chunking baseline changed. This interaction is intentionally preserved as an experimental result.

### Additional Experiments

The repository also contains experiments with:

- 15 fixed-size chunking configurations
- Parent-child retrieval
- Semantic context deduplication
- MMR context selection
- Multi-query retrieval
- HyDE-style retrieval
- Adaptive routing

Context clustering and MMR were rejected for the production path because they added substantial latency without improving evidence coverage on the current benchmark.

Multi-query retrieval was also rejected after failing to improve the Hybrid baseline while increasing retrieval latency.

Experimental implementations are kept as evidence even when they are not part of the production pipeline.

## Quick Start

Docker is the recommended way to run AstraRAG.

```bash
git clone <repository-url>
cd AstraRAG
```

Create `.env`:

```env
GEMINI_API_KEY=your_api_key
```

Place PDFs under:

```text
data/documents/
```

Build the image:

```bash
docker compose build
```

Index the documents:

```bash
docker compose run --rm astrarag astrarag index
```

Start the API:

```bash
docker compose up -d
```

Open the API documentation at:

`http://localhost:8000/docs`

Qdrant data is persisted in the `qdrant-data` Docker volume.

## Local Installation

AstraRAG requires Python 3.11 and `uv`.

```bash
uv sync
```

Download the pinned GGUF model and tokenizer:

```bash
mkdir -p models

curl -fL \
  https://huggingface.co/LLukas22/all-MiniLM-L6-v2-GGUF/resolve/ae6369f25f3393dcca54ff7b41157ffd2e43e2e6/all-minilm-l6-v2_f16.gguf \
  -o models/all-minilm-l6-v2_f16.gguf

mkdir -p models/all-MiniLM-L6-v2
curl -fL \
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/tokenizer.json \
  -o models/all-MiniLM-L6-v2/tokenizer.json

printf '%s  %s\n' \
  7266519c8c9b860e99d561d2cb849cf546b248eb23bcb23787be5b11180daec3 \
  models/all-minilm-l6-v2_f16.gguf | sha256sum -c -
```

Create `.env`:

```env
GEMINI_API_KEY=your_api_key
```

Start llama.cpp in a separate terminal:

```bash
llama serve \
  -m models/all-minilm-l6-v2_f16.gguf \
  --embedding --pooling mean --embd-normalize 2 \
  --host 127.0.0.1 --port 8081 --no-ui
```

Then index and run. Local file-backed Qdrant is intended for development; Docker Compose uses the Qdrant server so indexing and API processes can operate independently.

```bash
uv run astrarag index
uv run astrarag serve
```

A custom PDF directory can also be indexed:

```bash
uv run astrarag index /path/to/pdfs
```

## Python Usage

```python
from astrarag import AstraRAG

with AstraRAG() as rag:
    result = rag.query("What is the main idea behind Sentence-BERT?")

    print(result.generation.answer)
    print(result.route)
    print(result.retrieval_latency_ms)
```

## REST API

```text
GET  /health
POST /query
GET  /docs
```

Example:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main idea behind Sentence-BERT?"}'
```

Responses include the generated answer, selected retrieval route, source metadata, retrieval latency, and total pipeline latency.

## Evaluation

Retrieval experiments measure:

- Hit@5
- Hit@10
- MRR
- nDCG@10
- Retrieval latency

Development follows the same loop throughout the project:

```text
Baseline
   ↓
Benchmark
   ↓
Error Analysis
   ↓
Hypothesis
   ↓
Experiment
   ↓
Measure
   ↓
Keep / Reject
```

Failed and neutral experiments are retained rather than presented as improvements.

## Tests

```bash
uv run pytest
```

The test suite is split into `test/white_box` component tests and
`test/black_box` CLI/HTTP boundary tests. Slow local-model tests require
`RUN_SLOW_TESTS=1` and a llama.cpp server at `EMBEDDING_URL`; live external-service tests remain opt-in.

Lint:

```bash
uv run ruff check src test
```

## Tech Stack

Python 3.11 · uv · PyMuPDF · llama.cpp · Qdrant · rank-bm25 · FastAPI · Pydantic · Gemini API · pytest
