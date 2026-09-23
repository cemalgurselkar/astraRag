from google import genai
from google.genai import types

from astrarag.generation.prompt import (
    SYSTEM_INSTRUCTION,
    build_generation_prompt,
)
from astrarag.schemas import ContextPackage, GeneratedAnswer


class GeminiGenerator:
    def __init__(
        self,
        api_key: str,
        model: str,
    ) -> None:

        self.model = model
        self._client = genai.Client(api_key=api_key)

    def generate(
        self,
        query: str,
        context: ContextPackage,
    ) -> GeneratedAnswer:

        if not query.strip():
            raise ValueError("query must not be empty")

        if not context.items:
            return GeneratedAnswer(
                answer=(
                    "The available evidence is insufficient to answer this question."
                ),
                model=self.model,
                source_ids=[],
            )

        prompt = build_generation_prompt(
            query=query,
            context=context.text,
        )

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.0,
            ),
        )

        answer = response.text

        if not answer or not answer.strip():
            raise RuntimeError("Gemini returned an empty response.")

        return GeneratedAnswer(
            answer=answer.strip(),
            model=self.model,
            source_ids=[item.chunk_id for item in context.items],
        )

    def close(self) -> None:
        self._client.close()
