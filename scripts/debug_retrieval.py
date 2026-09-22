from pathlib import Path

from astrarag.embedding import EmbeddingEncoder
from astrarag.evaluation import load_evaluation_dataset
from astrarag.index import DenseVectorIndex
from astrarag.retrieval import DenseRetriever


QUERY_IDS = {
    "attention-006",
    "attention-008"
}

def main() -> None:
    
    encoder = EmbeddingEncoder()
    
    index = DenseVectorIndex(
        path=Path("data/qdrant"),
        collection_name="attention",
        dimension=encoder.dimension
    )
    
    retriever = DenseRetriever(
        encoder=encoder,
        index=index
    )
    
    dataset = load_evaluation_dataset(
        Path("data/eval/attention.json")
    )
    
    for eval_query in dataset:
        if eval_query.id not in QUERY_IDS:
            continue
            
        results = retriever.retriever(
            query=eval_query.query,
            top_k=10,
        )
        
        print("\n" + "=" * 100)
        print(f"QUERY: {eval_query.query}")
        print(f"EXPECTED: {eval_query.relevant_chunk_ids}")
        print("=" * 100)

        for rank, result in enumerate(results, start=1):
            relevant = result.chunk_id in eval_query.relevant_chunk_ids

            print(
                f"\n#{rank} "
                f"score={result.score:.4f} "
                f"page={result.page_numbers} "
                f"relevant={relevant}"
            )
            print(f"chunk={result.chunk_id}")
            print(result.text[:350].replace("\n", " "))

if __name__ == '__main__':
    main()