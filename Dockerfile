FROM python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336 /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md ./

RUN uv sync \
    --frozen \
    --no-dev \
    --no-install-project

COPY src ./src

RUN uv sync \
    --frozen \
    --no-dev

RUN uvx --from huggingface-hub hf download \
    sentence-transformers/all-MiniLM-L6-v2 \
    tokenizer.json \
    --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
    --local-dir /app/models/all-MiniLM-L6-v2


FROM python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce AS runtime

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
