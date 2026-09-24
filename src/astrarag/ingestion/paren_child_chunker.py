"""Create linked parent and child chunks for hierarchical retrieval."""

from astrarag.schemas import Chunk, Document


class ParentChildChunker:
    def __init__(
        self, parent_size: int = 2000, child_size: int = 500, child_overlap: int = 100
    ) -> None:

        if parent_size <= 0:
            raise ValueError("parent_size must be greater than 0")
        if child_size <= 0:
            raise ValueError("child_size must be greater than 0")

        if child_overlap < 0:
            raise ValueError("child_overlap cannot be negative")

        if child_overlap >= child_size:
            raise ValueError("child_overlap must be smaller than child_size")

        if child_size > parent_size:
            raise ValueError("child_size cannot be larger than parent_size")

        self.parent_size = parent_size
        self.child_size = child_size
        self.child_overlap = child_overlap

    def chunk(self, document: Document) -> tuple[list[Chunk], list[Chunk]]:

        parents: list[Chunk] = []
        children: list[Chunk] = []

        parent_index = 0
        child_index = 0

        for page in document.pages:
            text = page.text
            parent_start = 0

            while parent_start < len(text):
                parent_end = min(parent_start + self.parent_size, len(text))

                parent_text = text[parent_start:parent_end].strip()

                if not parent_text:
                    break

                parent_id = f"{document.id}:parent:{parent_index}"

                parent = Chunk(
                    id=parent_id,
                    document_id=document.id,
                    text=parent_text,
                    index=parent_index,
                    page_numbers=[page.number],
                    metadata={
                        **document.metadata,
                        "chunk_type": "parent",
                        "start_char": parent_start,
                        "end_char": parent_end,
                    },
                )

                parents.append(parent)

                child_start = parent_start

                while child_start < parent_end:
                    child_end = min(child_start + self.child_size, parent_end)

                    child_text = text[child_start:child_end].strip()

                    if child_text:
                        child = Chunk(
                            id=f"{document.id}:child:{child_index}",
                            document_id=document.id,
                            text=child_text,
                            index=child_index,
                            page_numbers=[page.number],
                            metadata={
                                **document.metadata,
                                "chunk_type": "child",
                                "parent_id": parent_id,
                                "parent_text": parent_text,
                                "parent_start_char": parent_start,
                                "parent_end_char": parent_end,
                                "start_char": child_start,
                                "end_char": child_end,
                            },
                        )

                        children.append(child)
                        child_index += 1

                    if child_end == parent_end:
                        break

                    child_start = child_end - self.child_overlap

                parent_index += 1
                parent_start = parent_end

        return parents, children
