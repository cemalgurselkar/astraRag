from astrarag.schemas import Chunk, Document

class ParentChildChunker:
    
    def __init__(self, parent_size: int=2000, child_size: int=500, child_overlap: int=100) -> None:
        pass