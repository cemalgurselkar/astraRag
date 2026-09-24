"""Create an initial annotated evaluation dataset from parsed and chunked PDFs."""

import json
from pathlib import Path

from astrarag.ingestion import FixedSizeChunker, PDFParser

QUERIES = [
    ("attention-001", "What is multi-head attention?", "semantic",
     "Multi-head attention allows the model"),

    ("attention-002", "What is scaled dot-product attention?", "semantic",
     "Scaled Dot-Product Attention"),

    ("attention-003", "How many attention heads are used in the Transformer?", "numerical",
     "we employ h = 8"),

    ("attention-004", "What is the dimensionality of the model embeddings?", "numerical",
     "dmodel = 512"),

    ("attention-005", "How many identical layers are used in the encoder?", "numerical",
     "stack of N = 6 identical layers"),

    ("attention-006", "Why does the Transformer use positional encodings?", "semantic",
     "Since our model contains no recurrence and no convolution"),

    ("attention-007", "What activation function is used in the position-wise feed-forward network?", "lexical",
     "ReLU"),

    ("attention-008", "What optimizer is used to train the Transformer?", "lexical",
     "Adam optimizer"),

    ("attention-009", "What dropout rate is used during training?", "numerical",
     "Pdrop = 0.1"),

    ("attention-010", "How does self-attention compare with recurrent layers in computational complexity?", "comparison",
     "Complexity per Layer"),

    ("attention-011", "What are the three ways multi-head attention is used in the Transformer?", "single_hop",
     "uses multi-head attention in three different ways"),

    ("attention-012", "Why was self-attention chosen for the Transformer architecture?", "semantic",
     "Self-attention is an attention mechanism"),
]


def main() -> None:
    parser = PDFParser()
    document = parser.parse(Path("data/raw/attention.pdf"))

    chunker = FixedSizeChunker(
        chunk_size=1000,
        overlap=200,
    )
    chunks = chunker.chunk(document)

    dataset = []

    for query_id, query, query_type, evidence_search in QUERIES:
        matches = [
            chunk
            for chunk in chunks
            if evidence_search.lower() in chunk.text.lower()
        ]

        if not matches:
            print(f"WARNING: no evidence found for {query_id}")
            continue

        dataset.append(
            {
                "id": query_id,
                "query": query,
                "relevant_chunk_ids": [
                    chunk.id for chunk in matches
                ],
                "query_type": query_type,
            }
        )

        print(f"{query_id}:")
        for chunk in matches:
            print(f"  {chunk.id} pages={chunk.page_numbers}")

    output_path = Path("data/eval/attention.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print(f"Wrote {len(dataset)} queries to {output_path}")


if __name__ == "__main__":
    main()
