import argparse
from dataclasses import asdict
from pathlib import Path

from doc_finder.bootstrap import (
    build_default_ingestion_service,
    build_default_search_service,
)
from doc_finder.schemas.search import TagSearchRequest, TagSearchResponse
from doc_finder.services.ingestion_service import IngestionSummary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run doc-finder ingestion and search.")
    # CLI 표면적은 작게 유지한다. 하나는 인덱싱, 하나는 검색만 담당한다.
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Search indexed images by tags.")
    search_parser.add_argument("--query", required=True, help="Search query.")
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Maximum number of results to return.",
    )
    search_parser.add_argument(
        "--review-status",
        default=None,
        help="Optional review status filter.",
    )

    ingest_parser = subparsers.add_parser("ingest", help="Index images from a directory.")
    ingest_parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing image assets to ingest.",
    )

    args = parser.parse_args(argv)

    if args.command == "search":
        # CLI 검색도 HTTP와 같은 서비스 레이어를 타도록 맞춘다.
        service = build_default_search_service()
        result = service.search(
            TagSearchRequest(
                query=args.query,
                top_k=args.top_k,
                review_status=args.review_status,
            )
        )
        print(result.model_dump())
        return

    # MVP에서는 배치 인덱싱을 HTTP job 없이 돌릴 수 있게 CLI를 우선 경로로 둔다.
    service = build_default_ingestion_service()
    result = service.ingest_directory(Path(args.image_dir))
    print(asdict(result))


if __name__ == "__main__":
    main()
