from pathlib import Path

from astrarag.embedding import EmbeddingEncoder
from astrarag.index import DenseVectorIndex
from astrarag.ingestion import FixedSizeChunker, PDFParser
from astrarag.retrieval import DenseRetriever

pdf_path = Path("data/raw/attention.pdf")

# Parse PDF
parser = PDFParser()
document = parser.parse(pdf_path)

# Chunk document
chunker = FixedSizeChunker(
    chunk_size=1000,
    overlap=200,
)
chunks = chunker.chunk(document)

# Create embeddings
encoder = EmbeddingEncoder()

embeddings = encoder.encode(
    [chunk.text for chunk in chunks]
)

# Store embeddings in Qdrant
index = DenseVectorIndex(
    path=Path("data/qdrant"),
    collection_name="attention",
    dimension=encoder.dimension,
)

index.add(
    chunks=chunks,
    embeddings=embeddings,
)

print(f"Document: {document.source}")
print(f"Chunks: {len(chunks)}")
print(f"Embeddings: {embeddings.shape}")
print("Indexed successfully.")

# Dense retrieval
retriever = DenseRetriever(
    encoder=encoder,
    index=index,
)

query = "What is multi-head attention?"

results = retriever.retriever(
    query=query,
    top_k=5,
)

print(f"\nQuery: {query}\n")

for rank, result in enumerate(results, start=1):
    print(f"Rank: {rank}")
    print(f"Score: {result.score:.4f}")
    print(f"Pages: {result.page_numbers}")
    print(f"Chunk ID: {result.chunk_id}")
    print(f"Text:\n{result.text[:500]}")
    print("-" * 80)