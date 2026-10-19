#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${PROJECT_DIR}/models"
GGUF_PATH="${MODEL_DIR}/all-minilm-l6-v2_f16.gguf"
TOKENIZER_PATH="${MODEL_DIR}/all-MiniLM-L6-v2/tokenizer.json"

GGUF_URL="https://huggingface.co/LLukas22/all-MiniLM-L6-v2-GGUF/resolve/ae6369f25f3393dcca54ff7b41157ffd2e43e2e6/all-minilm-l6-v2_f16.gguf"
GGUF_SHA256="7266519c8c9b860e99d561d2cb849cf546b248eb23bcb23787be5b11180daec3"
TOKENIZER_URL="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/tokenizer.json"
TOKENIZER_SHA256="be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"

require_command() {
    local command_name="$1"
    local install_hint="$2"

    if ! command -v "${command_name}" >/dev/null 2>&1; then
        printf 'Error: %s is required. %s\n' "${command_name}" "${install_hint}" >&2
        exit 1
    fi
}

file_sha256() {
    local path="$1"

    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "${path}" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "${path}" | awk '{print $1}'
    else
        printf 'Error: sha256sum or shasum is required to verify model files.\n' >&2
        exit 1
    fi
}

download_verified() {
    local label="$1"
    local url="$2"
    local target="$3"
    local expected_sha256="$4"
    local temporary_file="${target}.part"

    if [[ -f "${target}" ]] && [[ "$(file_sha256 "${target}")" == "${expected_sha256}" ]]; then
        printf '%s is already downloaded and verified.\n' "${label}"
        return
    fi

    printf 'Downloading %s...\n' "${label}"
    curl --fail --location --retry 3 --retry-delay 2 "${url}" --output "${temporary_file}"

    if [[ "$(file_sha256 "${temporary_file}")" != "${expected_sha256}" ]]; then
        rm -f -- "${temporary_file}"
        printf 'Error: %s SHA-256 verification failed.\n' "${label}" >&2
        exit 1
    fi

    mv -- "${temporary_file}" "${target}"
    printf '%s downloaded and verified.\n' "${label}"
}

require_command uv "Install it from https://docs.astral.sh/uv/getting-started/installation/."
require_command curl "Install curl with your operating system package manager."
require_command awk "Install awk with your operating system package manager."

cd "${PROJECT_DIR}"

printf 'Installing locked Python dependencies...\n'
uv sync --frozen

mkdir -p \
    "${MODEL_DIR}/all-MiniLM-L6-v2" \
    "${PROJECT_DIR}/data/documents" \
    "${PROJECT_DIR}/data/eval" \
    "${PROJECT_DIR}/data/qdrant"

download_verified "all-MiniLM-L6-v2 GGUF model" "${GGUF_URL}" "${GGUF_PATH}" "${GGUF_SHA256}"
download_verified "all-MiniLM-L6-v2 tokenizer" "${TOKENIZER_URL}" "${TOKENIZER_PATH}" "${TOKENIZER_SHA256}"

if [[ ! -f "${PROJECT_DIR}/.env" ]]; then
    cp "${PROJECT_DIR}/.env.example" "${PROJECT_DIR}/.env"
    printf 'Created .env from .env.example. Add your GEMINI_API_KEY before running AstraRAG.\n'
else
    printf '.env already exists; leaving it unchanged.\n'
fi

if command -v llama >/dev/null 2>&1; then
    printf 'llama.cpp CLI detected: %s\n' "$(command -v llama)"
elif command -v llama-server >/dev/null 2>&1; then
    printf 'llama.cpp server detected: %s\n' "$(command -v llama-server)"
else
    printf '\nWarning: llama.cpp was not found. Install it before local embedding inference:\n' >&2
    printf 'https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md\n' >&2
fi

printf '\nSetup complete.\n'
printf '1. Add GEMINI_API_KEY to .env.\n'
printf '2. Place PDF files in data/documents/.\n'
printf '3. Start llama.cpp using the command documented in README.md.\n'
printf '4. Run: uv run astrarag index\n'
printf '5. Run: uv run astrarag serve\n'
