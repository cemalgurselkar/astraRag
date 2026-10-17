"""Document chunking retained for parent-child retrieval experiments."""

from astrarag.models import Chunk, Document


def chunk_parent_child(
    document: Document,
    parent_size: int = 2000,
    child_size: int = 500,
    child_overlap: int = 100,
) -> tuple[list[Chunk], list[Chunk]]:
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be greater than 0")
    if child_overlap < 0 or child_overlap >= child_size:
        raise ValueError("child_overlap must be between 0 and child_size")
    if child_size > parent_size:
        raise ValueError("child_size cannot be larger than parent_size")

    parents: list[Chunk] = []
    children: list[Chunk] = []
    parent_index = 0
    child_index = 0

    for page in document.pages:
        parent_start = 0
        while parent_start < len(page.text):
            parent_end = min(parent_start + parent_size, len(page.text))
            parent_text = page.text[parent_start:parent_end].strip()
            if not parent_text:
                break

            parent_id = f"{document.id}:parent:{parent_index}"
            parents.append(
                Chunk(
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
            )

            child_start = parent_start
            while child_start < parent_end:
                child_end = min(child_start + child_size, parent_end)
                child_text = page.text[child_start:child_end].strip()
                if child_text:
                    children.append(
                        Chunk(
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
                    )
                    child_index += 1
                if child_end == parent_end:
                    break
                child_start = child_end - child_overlap

            parent_index += 1
            parent_start = parent_end

    return parents, children
