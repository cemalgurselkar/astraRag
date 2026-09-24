from google import genai
from google.genai import types


class GeminiQueryTransformer:
    def __init__(
        self,
        api_key: str,
        model: str,
    ) -> None:
        self.model = model
        self._client = genai.Client(api_key=api_key)

    def generate_queries(
        self,
        query: str,
        count: int = 3,
    ) -> list[str]:
        if not query.strip():
            raise ValueError("query cannot be empty")

        if count <= 0:
            raise ValueError("count must be greater than 0")

        prompt = (
            f"Generate {count} alternative search queries for the "
            "following academic literature retrieval query.\n"
            "Preserve the original meaning. Use different terminology "
            "that may appear in scientific papers.\n"
            "Return ONLY one query per line. No numbering.\n\n"
            f"Query: {query}"
        )

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
            ),
        )

        if not response.text:
            raise RuntimeError(
                "Gemini returned an empty query transformation."
            )

        queries = [
            line.strip()
            for line in response.text.splitlines()
            if line.strip()
        ]

        return queries[:count]

    def generate_hypothetical_document(
        self,
        query: str,
    ) -> str:
        if not query.strip():
            raise ValueError("query cannot be empty")

        prompt = (
            "Write a short hypothetical scientific passage that would "
            "directly answer the following question. "
            "Use terminology likely to appear in an academic paper. "
            "Return only the passage.\n\n"
            f"Question: {query}"
        )

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
            ),
        )

        if not response.text or not response.text.strip():
            raise RuntimeError(
                "Gemini returned an empty hypothetical document."
            )

        return response.text.strip()

    def close(self) -> None:
        self._client.close()