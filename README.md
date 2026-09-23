# AstraRAG

AstraRAG is a performance-oriented RAG engine for experimenting with retrieval strategies, context construction, and adaptive query routing.

Instead of assuming that every query should use the same retrieval pipeline, AstraRAG supports retrieval routes with different quality and latency characteristics and evaluates their behavior through reproducible benchmarks.

The system can be used directly as a Python package, through its CLI, or as an HTTP API.

## Architecture

```text
Documents
   │
   ▼
PDF Parser → Cleaner → Chunker
   │
   ▼
Embedding Encoder
   │
   ├─────────────────────┐
   ▼                     ▼
Qdrant Dense Index    BM25 Index
   │                     │
   └──────────┬──────────┘
              │
Query → Query Profiler → Router
              │
       ┌──────┼───────────┐
       ▼      ▼           ▼
     BM25   Hybrid   Hybrid + CE
     CHEAP  MEDIUM    EXPENSIVE
       └──────┼───────────┘
              ▼
        Context Engine
              │
              ▼
         LLM Generator
              │
              ▼
     Answer + Source Metadata
```

The outer pipeline is deterministic. The language model is used for grounded answer generation rather than retrieval orchestration.

## Features

AstraRAG currently includes:

- PDF parsing and text cleaning
- Fixed-size and parent-child chunking
- Local sentence-transformer embeddings
- Persistent Qdrant vector storage
- Dense and BM25 sparse retrieval
- Hybrid retrieval with Reciprocal Rank Fusion
- Cross-encoder reranking
- Parent-child retrieval experiments
- Query profiling and rule-based adaptive routing
- Context deduplication and budgeting
- Grounded generation with source metadata
- Retrieval evaluation and benchmarking
- Python, CLI, and REST API interfaces
- Docker deployment

Embedding and cross-encoder inference run locally. The reference generation adapter currently uses the Gemini API.

## Adaptive Retrieval

AstraRAG exposes three retrieval routes:

```text
CHEAP
└── BM25

MEDIUM
└── Dense + BM25
    └── Reciprocal Rank Fusion

EXPENSIVE
└── Dense + BM25
    └── Reciprocal Rank Fusion
        └── Cross-Encoder Reranking
```

A lightweight query profiler extracts query characteristics and a rule-based router selects a retrieval route.

Adaptive routing is treated as an experiment rather than an assumed improvement. Its behavior is measured against fixed retrieval strategies.

## Benchmark

The current development benchmark contains 40 evaluation queries over a five-paper academic corpus.

| Strategy | Hit@5 | Hit@10 | MRR | nDCG@10 | Mean Retrieval Latency |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.700 | 0.775 | 0.568 | 0.622 | 0.71 ms |
| Hybrid | 0.700 | 0.825 | 0.513 | 0.583 | 9.05 ms |
| Hybrid + Cross-Encoder | 0.750 | 0.800 | 0.663 | 0.683 | 668.87 ms |
| Adaptive Router | 0.700 | 0.800 | 0.559 | 0.611 | 181.30 ms |

Adaptive route distribution:

```text
CHEAP       62.5%
MEDIUM      15.0%
EXPENSIVE   22.5%
```

These are development results from the current corpus and evaluation set, not general RAG performance claims.

The current rule-based router reduces the use of expensive cross-encoder reranking compared with always using it, but does not outperform the BM25 baseline on retrieval quality. This result is intentionally preserved as part of the experimental record.

## Quick Start with Docker

Docker is the simplest way to run AstraRAG. The image installs the Python environment and downloads the embedding and cross-encoder models during the build.

Clone the repository:

```bash
git clone <repository-url>
cd AstraRAG
```

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key
```

Place PDF documents under:

```text
data/
└── documents/
    ├── paper_1.pdf
    ├── paper_2.pdf
    └── ...
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

The API is available at:

```text
http://localhost:8000
```

Interactive OpenAPI documentation:

```text
http://localhost:8000/docs
```

Qdrant data is persisted under `data/qdrant/`, so the index survives container recreation.

## Local Installation

AstraRAG requires Python 3.11 and uses `uv`.

```bash
uv sync
```

Download the local models:

```bash
mkdir -p models

uv run hf download sentence-transformers/all-MiniLM-L6-v2 \
  --local-dir models/all-MiniLM-L6-v2

uv run hf download cross-encoder/ms-marco-MiniLM-L6-v2 \
  --local-dir models/ms-marco-MiniLM-L6-v2
```

Create `.env`:

```env
GEMINI_API_KEY=your_api_key
```

Place PDFs in `data/documents/` and build the index:

```bash
uv run astrarag index
```

A custom document directory can also be supplied:

```bash
uv run astrarag index /path/to/pdfs
```

Start the API:

```bash
uv run astrarag serve
```

## Python Usage

AstraRAG can be used independently from the HTTP layer:

```python
from astrarag import AstraRAG

with AstraRAG() as rag:
    result = rag.query(
        "What is the main idea behind Sentence-BERT?"
    )

    print(result.generation.answer)
    print(result.route)
    print(result.retrieval_latency_ms)
```

## REST API

The service exposes:

```text
GET  /health
POST /query
GET  /docs
```

Example request:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main idea behind Sentence-BERT?"
  }'
```

Example response:

```json
{
  "answer": "...",
  "route": "cheap",
  "sources": [
    {
      "chunk_id": "...",
      "document_id": "...",
      "pages": [3]
    }
  ],
  "retrieval_latency_ms": 0.79,
  "total_latency_ms": 8223.48
}
```

Retrieval latency is reported separately from total pipeline latency because generation currently depends on an external LLM provider.

## Evaluation

Retrieval strategies are evaluated independently from generation using:

```text
Hit@5
Hit@10
MRR
nDCG@10
Retrieval latency
```

The repository contains experiments for dense retrieval, BM25, hybrid retrieval, cross-encoder reranking, parent-child retrieval, and adaptive routing.

AstraRAG follows an experiment-driven development loop:

```text
Baseline
   ↓
Benchmark
   ↓
Error Analysis
   ↓
Identify Bottleneck
   ↓
Experiment
   ↓
Measure
   ↓
Keep / Reject
```

Failed or neutral experiments are useful evidence and are not presented as improvements.

## Tests

Run the unit test suite with:

```bash
uv run pytest test/unit -v
```

## Project Direction

AstraRAG establishes a measurable RAG system before introducing additional complexity.

The current router, chunking configuration, embedding model, and retrieval configuration are baselines rather than claims of optimality. Future experiments can investigate larger corpora, chunking and embedding ablations, improved context engineering, evidence grading, additional retrieval strategies, and retrieval-history-based routing.

## Tech Stack

Python 3.11 · uv · PyMuPDF · sentence-transformers · Qdrant · rank-bm25 · FastAPI · Pydantic · Gemini API · pytest