from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from doc_finder.repositories.image_index import ImageDocument, RejectedAsset
from doc_finder.services.embedding_service import HashingEmbeddingService
from doc_finder.services.filename_parser import FilenameParseError, parse_asset_filename
from doc_finder.services.image_metadata import ImageValidationError, read_image_metadata
from doc_finder.services.query_normalizer import QueryNormalizer
from doc_finder.services.tagging_service import TaggingError


@dataclass(slots=True)
class IngestionSummary:
    scanned_count: int = 0
    indexed_count: int = 0
    duplicate_count: int = 0
    reject_count: int = 0


class IngestionService:
    def __init__(
        self,
        repository,
        tagger,
        embedding_service: HashingEmbeddingService,
        query_normalizer: QueryNormalizer | None = None,
        max_file_size_bytes: int = 1_000_000,
        review_threshold: float = 0.8,
        progress_reporter: Callable[[str], None] | None = None,
    ) -> None:
        self._repository = repository
        self._tagger = tagger
        self._embedding_service = embedding_service
        self._query_normalizer = query_normalizer or QueryNormalizer()
        self._max_file_size_bytes = max_file_size_bytes
        self._review_threshold = review_threshold
        self._progress_reporter = progress_reporter
        self._repository.ensure_schema()

    def ingest_directory(self, directory: Path | str) -> IngestionSummary:
        summary = IngestionSummary()
        root = Path(directory)

        for asset_path in sorted(path for path in root.rglob("*") if path.is_file()):
            summary.scanned_count += 1
            try:
                # 무거운 작업 전에 파일명에서 저장 키를 먼저 뽑는다.
                parsed = parse_asset_filename(asset_path.name)
                metadata = read_image_metadata(
                    asset_path,
                    max_file_size_bytes=self._max_file_size_bytes,
                )
                if self._repository.has_sha256(metadata.sha256):
                    summary.duplicate_count += 1
                    self._report(
                        f"[duplicate] {asset_path} sha256={metadata.sha256}"
                    )
                    continue

                tagging_result = self._tagger.tag(asset_path, metadata.sha256)
                # exact 검색과 semantic 검색을 같이 쓰기 위해 정규화 태그와 projection 텍스트를 함께 저장한다.
                normalized_tags = self._normalize_tags(
                    tagging_result.keyword_tags,
                    tagging_result.normalized_tags,
                )
                tag_text_projection = self._build_tag_text_projection(
                    tagging_result.keyword_tags,
                    normalized_tags,
                )
                review_status = self._resolve_review_status(
                    tagging_result.review_status,
                    tagging_result.confidence,
                )
                inserted = self._repository.upsert_image(
                    ImageDocument(
                        unit_id=parsed.unit_id,
                        data_id=parsed.data_id,
                        image_id=parsed.image_id,
                        asset_path=str(asset_path),
                        sha256=metadata.sha256,
                        file_size=metadata.file_size,
                        width=metadata.width,
                        height=metadata.height,
                        keyword_tags=tagging_result.keyword_tags,
                        normalized_tags=normalized_tags,
                        tag_text_projection=tag_text_projection,
                        confidence=tagging_result.confidence,
                        review_status=review_status,
                        embedding=self._embedding_service.embed_text(tag_text_projection),
                        tagged_at=datetime.now(UTC),
                    )
                )
                if inserted:
                    summary.indexed_count += 1
                    self._report(
                        "[indexed] "
                        f"{asset_path} "
                        f"keyword_tags={tagging_result.keyword_tags} "
                        f"normalized_tags={normalized_tags} "
                        f"confidence={tagging_result.confidence:.2f} "
                        f"review_status={review_status}"
                    )
                else:
                    summary.duplicate_count += 1
                    self._report(
                        f"[duplicate] {asset_path} sha256={metadata.sha256}"
                    )
            except FilenameParseError as exc:
                # reject 이력은 남겨야 나중에 잘못된 파일만 따로 고쳐서 재처리할 수 있다.
                self._record_reject(summary, asset_path, "invalid_filename", str(exc))
            except ImageValidationError as exc:
                self._record_reject(summary, asset_path, exc.reason, str(exc))
            except TaggingError as exc:
                self._record_reject(summary, asset_path, "tagging_failed", str(exc))

        return summary

    def _normalize_tags(
        self,
        keyword_tags: list[str],
        normalized_tags: list[str],
    ) -> list[str]:
        source_tags = normalized_tags or keyword_tags
        unique_tags = {self._query_normalizer.normalize_tag(tag) for tag in source_tags}
        return sorted(tag for tag in unique_tags if tag)

    def _build_tag_text_projection(
        self,
        keyword_tags: list[str],
        normalized_tags: list[str],
    ) -> str:
        parts = list(
            dict.fromkeys(
                [
                    *self._query_normalizer.projection_terms(normalized_tags),
                    *keyword_tags,
                ]
            )
        )
        return " ".join(parts)

    def _resolve_review_status(
        self,
        requested_review_status: str,
        confidence: float,
    ) -> str:
        # 이미 승인 외 상태가 정해져 있지 않을 때만 confidence 기준으로 pending 처리한다.
        if requested_review_status != "approved":
            return requested_review_status
        if confidence < self._review_threshold:
            return "pending"
        return "approved"

    def _record_reject(
        self,
        summary: IngestionSummary,
        asset_path: Path,
        reason: str,
        detail: str,
    ) -> None:
        self._repository.record_reject(
            RejectedAsset(
                asset_path=str(asset_path),
                reason=reason,
                detail=detail,
            )
        )
        summary.reject_count += 1
        self._report(f"[reject] {asset_path} reason={reason} detail={detail}")

    def _report(self, message: str) -> None:
        if self._progress_reporter is not None:
            self._progress_reporter(message)
