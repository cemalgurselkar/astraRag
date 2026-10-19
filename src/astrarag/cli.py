"""PDF indeksleme, API sunma ve değerlendirme girişi için CLI sağlar.

Sözlük:
- `build_parser`: `index`, `serve` ve `evaluate` komutlarını tanımlar.
- `main`: Argümanları okuyup ilgili komutu çalıştırır.
- `run_index`: PDF dizinini artımlı olarak production koleksiyonuna indeksler.
- `run_server`: FastAPI uygulamasını Uvicorn ile başlatır.
- `run_evaluation`: Henüz taslak olan değerlendirme komutunun veri setini gösterir.
"""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="astrarag",
        description="AstraRAG command line interface.",
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    index_parser = commands.add_parser(
        "index",
        help="Index PDF documents.",
    )
    index_parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("data/documents"),
        help="PDF directory. Default: data/documents",
    )

    commands.add_parser(
        "serve",
        help="Start the AstraRAG HTTP API.",
    )

    evaluate_parser = commands.add_parser(
        "evaluate",
        help="Run retrieval evaluation.",
    )
    evaluate_parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("data/eval/evaluation.json"),
        help="Evaluation dataset.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "index":
        run_index(args.path)

    elif args.command == "serve":
        run_server()

    elif args.command == "evaluate":
        run_evaluation(args.path)


def run_index(path: Path) -> None:
    from astrarag.config import Settings
    from astrarag.indexing import index_directory

    settings = Settings()

    try:
        document_count = index_directory(
            path,
            settings=settings,
        )
    except (
        FileNotFoundError,
        NotADirectoryError,
        ValueError,
        RuntimeError,
    ) as exc:
        raise SystemExit(f"Indexing failed: {exc}") from exc

    print(f"Successfully indexed {document_count} document(s).")


def run_server() -> None:
    import uvicorn

    uvicorn.run(
        "astrarag.api.app:app",
        host="0.0.0.0",
        port=8000,
    )


def run_evaluation(path: Path) -> None:
    print(f"Evaluation dataset: {path}")


if __name__ == "__main__":
    main()
