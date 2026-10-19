# AstraRAG

AstraRAG is a small end-to-end RAG project for indexing PDFs and answering
questions with cited sources. It uses local embeddings, Qdrant retrieval, and
Gemini only for answer generation.

## Design

```text
PDF files
   │
   ▼
Parse → token chunks → llama.cpp embeddings → Qdrant HNSW
                                               │
Query → simple router ──┬─ BM25 ───────────────┤
                       └─ Dense + BM25 (RRF) ──┘
                                               │
                                               ▼
                                      Context → Gemini → Answer
```

The query router selects fast BM25 for simple queries and Hybrid retrieval for
queries that need broader matching. Retrieval is deterministic; Gemini only
receives the selected context.

Main behavior:

- `all-MiniLM-L6-v2` embeddings run through llama.cpp without Torch.
- Qdrant uses HNSW ANN search; BM25 runs in memory.
- Index updates are atomic. A failed build does not replace the active index.
- Re-indexing skips unchanged corpora and reuses vectors for unchanged PDFs.
- Pydantic validates settings, documents, retrieval results, and API responses.
- Parent-child, MMR, clustering, MultiQuery, and HyDE remain optional experiments.

See [architecture.md](architecture.md) for the file map and implementation
notes.

## Quick Start

Requirements: Python 3.11+, `uv`, `curl`, and a llama.cpp installation.

```bash
git clone https://github.com/cemalgurselkar/astraRag.git
cd astraRag
./setup.sh
```

`setup.sh` installs the locked dependencies and downloads the pinned GGUF model
and tokenizer. It warns if llama.cpp is not installed.

Add a Gemini API key to `.env`:

```env
GEMINI_API_KEY=your_api_key
```

Put PDF files in `data/documents/`, then start the embedding server:

```bash
llama serve \
  -m models/all-minilm-l6-v2_f16.gguf \
  --embedding --pooling mean --embd-normalize 2 \
  --host 127.0.0.1 --port 8081 --no-ui
```

In another terminal, build the index:

```bash
uv run astrarag index
```

Use a different PDF directory when needed:

```bash
uv run astrarag index /path/to/pdfs
```

Repeated indexing is incremental. An unchanged corpus does not call the PDF
parser or embedding server. Model, chunk, or index-schema changes trigger one
safe full rebuild.

## Usage

Choose either the terminal demo or the HTTP API. When using the default local
file-backed Qdrant, they cannot run at the same time because only one process
can open `data/qdrant`.

### Terminal

Run the terminal demo and enter a question:

```bash
uv run python main.py
```

You can also pass the question directly:

```bash
uv run python main.py "What is the main idea behind Sentence-BERT?"
```

The terminal prints the generated answer, selected retrieval route, sources,
and latency measurements. It also prepares the index automatically: a missing
or incompatible index is rebuilt, changed PDFs are updated, and an unchanged
index is reused without calling the embedding model.

### HTTP API

Start the API:

```bash
uv run astrarag serve
```

OpenAPI documentation is available at `http://localhost:8000/docs`.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the main idea behind Sentence-BERT?"}'
```

The response includes the answer, selected route, sources, retrieval time, and
total time.

Python usage:

```python
from astrarag import AstraRAG

with AstraRAG() as rag:
    result = rag.query("What is the main idea behind Sentence-BERT?")
    print(result.generation.answer)
    print(result.context.items)
```

## Docker

Docker Compose runs Qdrant, llama.cpp, and AstraRAG as separate services:

```bash
docker compose build
docker compose run --rm astrarag astrarag index
docker compose up -d
```

The Qdrant index is stored in the `qdrant-data` volume. PDFs are read from
`data/documents/`.

## Performance

The included benchmark uses 40 queries, five academic PDFs, and 314 chunks.
Measurements are warm runs on the development machine, so absolute latency
will vary by hardware.

| Retrieval | Hit@5 | Hit@10 | MRR | nDCG@10 | Mean | p95 |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 0.525 | 0.625 | 0.394 | 0.440 | 7.78 ms | 9.37 ms |
| BM25 | **0.700** | **0.775** | **0.556** | **0.603** | **0.71 ms** | **1.15 ms** |
| Hybrid RRF | 0.675 | 0.750 | 0.525 | 0.564 | 9.38 ms | 12.42 ms |

The adaptive router reached **0.725 Hit@5** at **4.30 ms mean retrieval
latency**. It selected BM25 for 62.5% and Hybrid for 37.5% of the queries.

Warm component latency:

| Component | Mean | p95 |
|---|---:|---:|
| llama.cpp query embedding | 5.31 ms | 7.01 ms |
| Qdrant ANN search | 2.57 ms | 3.74 ms |
| BM25 search | 0.97 ms | 1.43 ms |

Incremental indexing reduced an unchanged five-PDF run from about **12.4 s**
to **1.3 s** by avoiding PDF parsing and embedding.

Run the benchmarks with:

```bash
uv run python scripts/benchmark_retrieval.py
uv run python scripts/benchmark_router.py
uv run python scripts/benchmark_chunking.py
```

## Tests

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

`test/white_box` contains component tests. `test/black_box` covers CLI, HTTP,
indexing, and end-to-end boundaries. Tests requiring llama.cpp or external
services are opt-in through their documented environment variables.

## Current Limits

- Text-based PDFs are supported; OCR and complex table/layout extraction are not.
- BM25 is rebuilt in memory when the application starts.
- `astrarag evaluate` is currently a placeholder; benchmark scripts are the
  working evaluation entry points.

## License

[MIT](LICENSE)

## Example Output

```text
$ uv run python main.py
Question: What is Transformer

Checking document index...
Index is already up to date: 5 documents, 314 chunks.
Starting AstraRAG...

Route: cheap
Retrieval: 0.47 ms
Total: 12841.19 ms

Answer
------
Based on the provided context, the Transformer is a model architecture that
uses stacked self-attention and point-wise, fully connected layers for both
its encoder and decoder [SOURCE 8].

Sources
-------
1. attention.pdf | pages: 14 | score: 9.5978
2. attention.pdf | pages: 15 | score: 9.5978
3. CoIBERT.pdf | pages: 6 | score: 6.7712
...
```

This query used the `cheap` BM25 route, so llama.cpp embedding and Qdrant ANN
search were not part of the query path. Retrieval took 0.47 ms; most of the
total time was Gemini API generation. End-to-end latency varied between roughly
3 and 13 seconds during local runs because external API and network latency are
variable.
