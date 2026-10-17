"""Exercise Gemini answer generation with a small synthetic retrieval context."""

from astrarag.config import Settings
from astrarag.context import build_context
from astrarag.generation import GeminiGenerator
from astrarag.models import RetrievalResult


def main() -> None:
    settings = Settings()
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is required for this smoke test")

    results = [
        RetrievalResult(
            chunk_id="test-1",
            document_id="attention-paper",
            text=(
                "The Transformer is a model architecture "
                "eschewing recurrence and instead relying "
                "entirely on an attention mechanism."
            ),
            score=1.0,
            page_numbers=[1],
            metadata={
                "filename": "attention.pdf",
            },
        )
    ]

    context = build_context(results)

    generator = GeminiGenerator(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
    )

    try:
        answer = generator.generate(
            query=("What does the Transformer rely on instead of recurrence?"),
            context=context,
        )

        print("\nAnswer:")
        print(answer.answer)

        print("\nModel:")
        print(answer.model)

        print("\nRetrieved source IDs:")
        print(answer.source_ids)

    finally:
        generator.close()


if __name__ == "__main__":
    main()
