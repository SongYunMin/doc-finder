from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from doc_finder.services.filename_parser import FilenameParseError, parse_asset_filename
from doc_finder.services.image_metadata import ImageValidationError, read_image_metadata
from doc_finder.services.tagging_service import TaggingError


@dataclass(slots=True)
class TagPreviewRejection:
    asset_path: str
    reason: str
    detail: str


@dataclass(slots=True)
class TagPreviewSummary:
    scanned_count: int = 0
    tagged_count: int = 0
    reject_count: int = 0
    rejections: list[TagPreviewRejection] = field(default_factory=list)


@dataclass(slots=True)
class TagPreviewResult:
    asset_path: str
    od_raw: list[str] | None = None
    ocr_raw: str | None = None
    describe_raw: str | None = None


@dataclass(slots=True)
class TagPreviewReport:
    summary: TagPreviewSummary
    results: list[TagPreviewResult]


class TagPreviewService:
    def __init__(
        self,
        tagger,
        max_file_size_bytes: int = 1_000_000,
    ) -> None:
        self._tagger = tagger
        self._max_file_size_bytes = max_file_size_bytes

    def preview_directory(self, directory: Path | str) -> TagPreviewReport:
        summary = TagPreviewSummary()
        results: list[TagPreviewResult] = []
        root = Path(directory)

        # 저장 경로를 타지 않고 ingest 전단의 검증/태깅 정책만 재사용한다.
        for asset_path in sorted(path for path in root.rglob("*") if path.is_file()):
            summary.scanned_count += 1
            try:
                parse_asset_filename(asset_path.name)
                metadata = read_image_metadata(
                    asset_path,
                    max_file_size_bytes=self._max_file_size_bytes,
                )
                preview_method = getattr(self._tagger, "preview_raw", None)
                if preview_method is None:
                    raise TaggingError("raw preview is not supported by this tagger.")

                preview_result = preview_method(asset_path, metadata.sha256)
                results.append(
                    TagPreviewResult(
                        asset_path=str(asset_path),
                        od_raw=(
                            list(preview_result.od_raw)
                            if preview_result.od_raw is not None
                            else None
                        ),
                        ocr_raw=(
                            str(preview_result.ocr_raw)
                            if preview_result.ocr_raw is not None
                            else None
                        ),
                        describe_raw=(
                            str(preview_result.describe_raw)
                            if preview_result.describe_raw is not None
                            else None
                        ),
                    )
                )
                summary.tagged_count += 1
            except FilenameParseError as exc:
                self._record_reject(summary, asset_path, "invalid_filename", str(exc))
            except ImageValidationError as exc:
                self._record_reject(summary, asset_path, exc.reason, str(exc))
            except TaggingError as exc:
                self._record_reject(summary, asset_path, "tagging_failed", str(exc))

        return TagPreviewReport(summary=summary, results=results)

    def _record_reject(
        self,
        summary: TagPreviewSummary,
        asset_path: Path,
        reason: str,
        detail: str,
    ) -> None:
        summary.reject_count += 1
        summary.rejections.append(
            TagPreviewRejection(
                asset_path=str(asset_path),
                reason=reason,
                detail=detail,
            )
        )
