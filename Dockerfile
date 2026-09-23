FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md ./

RUN uv sync \
    --frozen \
    --no-dev \
    --no-install-project

COPY src ./src

RUN uv sync \
    --frozen \
    --no-dev

# Models are downloaded once while building the image.
RUN uv run hf download \
    sentence-transformers/all-MiniLM-L6-v2 \
    --local-dir /app/models/all-MiniLM-L6-v2

RUN uv run hf download \
    cross-encoder/ms-marco-MiniLM-L6-v2 \
    --local-dir /app/models/ms-marco-MiniLM-L6-v2


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HUB_OFFLINE=1

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/models /app/models

RUN mkdir -p \
    /app/data/documents \
    /app/data/eval \
    /app/data/qdrant

EXPOSE 8000

CMD ["uvicorn", "astrarag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]