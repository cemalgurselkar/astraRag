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
Embedding Encoder
   │
   ├───────────────┐
   ▼               ▼
Qdrant          BM25 Index
   │               │
   └───────┬───────┘
           │
Query → Profiler → Router
           │
     ┌─────┼──────────┐
     ▼     ▼          ▼
   BM25  Hybrid   Hybrid + CE
     │     │          │
     └─────┴──────────┘
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
- Local sentence-transformer embeddings
- Persistent Qdrant vector storage
- Dense and BM25 retrieval
- Hybrid retrieval with Reciprocal Rank Fusion
- Cross-encoder reranking
- Parent-child retrieval
- Rule-based adaptive routing
- Context deduplication and budgeting
- Gemini-based grounded generation
- Retrieval and context benchmarking
- Python, CLI, REST API, and Docker interfaces

Embedding and cross-encoder inference run locally. Only generation requires the external Gemini API.

## Experiments

AstraRAG uses a 40-query development benchmark over five academic papers.

The current fixed-size chunking baseline is **1800 characters / 200 overlap**, selected through a 15-configuration chunking ablation.

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

Qdrant data is persisted under `data/qdrant/`.

## Local Installation

AstraRAG requires Python 3.11 and `uv`.

```bash
uv sync
```

Download the local models:

```bash
mkdir -p models

uv run hf download sentence-transformers/all-MiniLM-L6-v2 \
  --local-dir models/all-MiniLM-L6-v2

uv run hf download cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --local-dir models/ms-marco-MiniLM-L6-v2
```

Create `.env`:

```env
GEMINI_API_KEY=your_api_key
```

Then index and run:

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
    result = rag.query(
        "What is the main idea behind Sentence-BERT?"
    )

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
uv run pytest test/unit -v
```

Lint:

```bash
uv run ruff check src test
```

## Tech Stack

Python 3.11 · uv · PyMuPDF · sentence-transformers · Qdrant · rank-bm25 · FastAPI · Pydantic · Gemini API · pytest