"""Getirilen bağlamı kullanarak Gemini ile kaynaklandırılmış cevap üretir.

Sözlük:
- `SYSTEM_INSTRUCTION`: Modeli yalnız verilen kanıtlara bağlı tutan kurallar.
- `GeminiGenerator`: Gemini istemcisini, geçici hata tekrarlarını ve cevabı yönetir.
"""

from time import sleep

from google import genai
from google.genai import errors, types

from astrarag.models import ContextPackage, GeneratedAnswer

SYSTEM_INSTRUCTION = """
You are the generation component of AstraRAG, an academic retrieval-augmented
generation system.

Answer the user's question using only the provided retrieved context.

Rules:
- Do not use outside knowledge.
- If the context does not contain enough evidence, say that the available
  evidence is insufficient.
- Do not invent facts.
- Cite supporting sources using [SOURCE N].
- Every factual claim should be supported by the provided context.
- Prefer a concise, direct answer.
""".strip()


class GeminiGenerator:
    """Own a Gemini client and generate answers from packaged context."""

    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self._client = genai.Client(api_key=api_key)

    def generate(self, query: str, context: ContextPackage) -> GeneratedAnswer:
        if not query.strip():
            raise ValueError("query must not be empty")
        if not context.items:
            return GeneratedAnswer(
                answer="The available evidence is insufficient to answer this question.",
                model=self.model,
                source_ids=[],
            )

        prompt = f"Retrieved context:\n\n{context.text}\n\nQuestion:\n{query.strip()}"
        response = None
        for attempt in range(3):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.0,
                        automatic_function_calling=(
                            types.AutomaticFunctionCallingConfig(disable=True)
                        ),
                    ),
                )
                break
            except errors.ServerError as exc:
                if attempt == 2:
                    raise RuntimeError(
                        f"Gemini is temporarily unavailable after 3 attempts: {exc}"
                    ) from exc
                sleep(2**attempt)
            except errors.APIError as exc:
                raise RuntimeError(f"Gemini request failed: {exc}") from exc

        if response is None:
            raise RuntimeError("Gemini returned no response")
        if not response.text or not response.text.strip():
            raise RuntimeError("Gemini returned an empty response")

        return GeneratedAnswer(
            answer=response.text.strip(),
            model=self.model,
            source_ids=[item.chunk_id for item in context.items],
        )

    def close(self) -> None:
        self._client.close()
