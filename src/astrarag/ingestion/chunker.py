"""Split documents into overlapping fixed-size chunks with source metadata."""

from astrarag.schemas import Chunk, Document


class FixedSizeChunker:
    def __init__(self, chunk_size: int = 1800, overlap: int = 200):

        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0")
        if overlap < 0:
            raise ValueError("Chunk overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, documents: Document) -> list[Chunk]:

        chunks: list[Chunk] = []
        chunk_index = 0

        for page in documents.pages:
            text = page.text
            start = 0

            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunks.append(
                        Chunk(
                            id=f"{documents.id}:{chunk_index}",
                            document_id=documents.id,
                            text=chunk_text,
                            index=chunk_index,
                            page_numbers=[page.number],
                            metadata={
                                **documents.metadata,
                                "start_char": start,
                                "end_char": end,
                            },
                        )
                    )

                    chunk_index += 1
                    if end == len(text):
                        break

                    start = end - self.overlap

        return chunks
