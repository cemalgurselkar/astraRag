"""AstraRAG'i terminalden çalıştıran küçük bir uçtan uca örnek.

Sözlük:
- `main`: İndeksi hazırlar, soruyu alır ve AstraRAG pipeline'ını çalıştırır.
- `_print_result`: Cevap, route, kaynak ve süreleri terminalde gösterir.
"""

import argparse

from astrarag import AstraRAG
from astrarag.config import Settings
from astrarag.indexing import index_directory
from astrarag.models import PipelineResult


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question over the indexed AstraRAG PDF corpus.",
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Question to ask. When omitted, it is read from the terminal.",
    )
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt) as exc:
            raise SystemExit("\nNo question provided.") from exc
    if not question:
        parser.error("question cannot be empty")

    settings = Settings()
    print("\nChecking document index...")
    try:
        index_directory(settings.documents_path, settings=settings)
        print("Starting AstraRAG...")
        with AstraRAG(settings) as rag:
            result = rag.query(question)
    except RuntimeError as exc:
        if "already accessed by another instance" in str(exc):
            raise SystemExit(
                "AstraRAG failed: local Qdrant storage is already in use.\n"
                "Stop the running 'astrarag serve' process before using main.py, "
                "or configure QDRANT_URL to use a Qdrant server."
            ) from exc
        raise SystemExit(f"AstraRAG failed: {exc}") from exc
    except (OSError, ValueError) as exc:
        raise SystemExit(f"AstraRAG failed: {exc}") from exc

    _print_result(result)


def _print_result(result: PipelineResult) -> None:
    print(f"\nRoute: {result.route.value}")
    print(f"Retrieval: {result.retrieval_latency_ms:.2f} ms")
    print(f"Total: {result.total_latency_ms:.2f} ms")
    print(f"\nAnswer\n------\n{result.generation.answer}")

    if not result.context.items:
        print("\nSources\n-------\nNo sources returned.")
        return

    print("\nSources\n-------")
    for number, item in enumerate(result.context.items, start=1):
        filename = item.metadata.get("filename", item.document_id)
        pages = ", ".join(str(page) for page in item.page_numbers) or "unknown"
        print(
            f"{number}. {filename} | pages: {pages} | "
            f"score: {item.score:.4f}"
        )


if __name__ == "__main__":
    main()
