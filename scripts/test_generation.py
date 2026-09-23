from astrarag.config import Settings
from astrarag.context import ContextEngine
from astrarag.generation import GeminiGenerator
from astrarag.schemas import RetrievalResult


def main() -> None:
    settings = Settings()

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

    context_engine = ContextEngine()
    context = context_engine.build(results)

    generator = GeminiGenerator(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
    )

    try:
        answer = generator.generate(
            query=(
                "What does the Transformer rely on "
                "instead of recurrence?"
            ),
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