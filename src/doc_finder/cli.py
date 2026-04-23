import argparse
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path

from doc_finder.bootstrap import (
    build_default_ingestion_service,
    build_default_search_service,
)
from doc_finder.schemas.search import TagSearchRequest, TagSearchResponse
from doc_finder.services.ingestion_service import IngestionSummary
from doc_finder.services.tag_preview_service import TagPreviewService
from doc_finder.taggers import build_tagger


def _drop_none_fields(value):
    if is_dataclass(value):
        return _drop_none_fields(asdict(value))
    if isinstance(value, dict):
        return {
            key: _drop_none_fields(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_drop_none_fields(item) for item in value]
    return value


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

    tag_parser = subparsers.add_parser(
        "tag",
        help="Preview tags from a directory without writing to the database.",
    )
    tag_parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing image assets to preview.",
    )
    tag_parser.add_argument(
        "--tagger-provider",
        default="florence2",
        help="Tagger provider to use for preview.",
    )
    tag_parser.add_argument(
        "--florence2-model-id",
        default=None,
        help="Optional Florence-2 model id override for preview runs.",
    )
    tag_parser.add_argument(
        "--paligemma2-model-id",
        default=None,
        help="Optional PaliGemma 2 model id override for preview runs.",
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

    if args.command == "tag":
        # DB 저장 없이 provider 실행 결과만 비교할 수 있도록 CLI override env를 만든다.
        runtime_environ = dict(os.environ)
        runtime_environ["DOC_FINDER_TAGGER_PROVIDER"] = args.tagger_provider
        if args.florence2_model_id is not None:
            runtime_environ["DOC_FINDER_FLORENCE2_MODEL_ID"] = args.florence2_model_id
        if args.paligemma2_model_id is not None:
            runtime_environ["DOC_FINDER_PALIGEMMA2_MODEL_ID"] = args.paligemma2_model_id

        tagger = build_tagger(
            args.tagger_provider,
            environ=runtime_environ,
        )
        report = TagPreviewService(tagger).preview_directory(Path(args.image_dir))
        # 사람이 비교하기 쉬운 진단 출력이 우선이므로 줄바꿈된 JSON으로 렌더링한다.
        print(json.dumps(_drop_none_fields(report), ensure_ascii=False, indent=2))
        return

    # MVP에서는 배치 인덱싱을 HTTP job 없이 돌릴 수 있게 CLI를 우선 경로로 둔다.
    service = build_default_ingestion_service(progress_reporter=print)
    result = service.ingest_directory(Path(args.image_dir))
    print(asdict(result))


if __name__ == "__main__":
    main()
