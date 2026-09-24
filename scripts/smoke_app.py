"""Run a minimal end-to-end query through the high-level AstraRAG interface."""

from astrarag import AstraRAG


def main() -> None:
    with AstraRAG() as rag:
        result = rag.query(
            "What is the main idea behind Sentence-BERT?"
        )

        print("\nAnswer:")
        print(result.generation.answer)

        print("\nRoute:")
        print(result.route.value)

        print("\nRetrieval latency:")
        print(f"{result.retrieval_latency_ms:.2f} ms")

        print("\nTotal latency:")
        print(f"{result.total_latency_ms:.2f} ms")

        print("\nSources:")
        for item in result.context.items:
            print(
                f"- {item.document_id} "
                f"{item.page_numbers}"
            )


if __name__ == "__main__":
    main()
