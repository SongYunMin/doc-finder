from pathlib import Path

from doc_finder.services.tag_preview_service import TagPreviewService
from doc_finder.services.tagging_service import TaggingError, TaggingResult


def _write_svg(path: Path, label: str) -> None:
    path.write_text(
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="120" height="80">'
            f"<text>{label}</text>"
            "</svg>"
        ),
        encoding="utf-8",
    )


class _ScriptedVisionTagger:
    def __init__(self, mapping: dict[str, TaggingResult], failures: set[str] | None = None) -> None:
        self._mapping = mapping
        self._failures = failures or set()

    def tag(self, asset_path: Path, sha256: str) -> TaggingResult:
        del sha256
        if asset_path.name in self._failures:
            raise TaggingError(f"failed for {asset_path.name}")
        return self._mapping[asset_path.name]


def test_tag_preview_service_builds_results_for_valid_directory(tmp_path: Path) -> None:
    first = tmp_path / "10565_20077_2.svg"
    second = tmp_path / "10565_20077_1.svg"
    _write_svg(first, "bus")
    _write_svg(second, "apple")

    service = TagPreviewService(
        tagger=_ScriptedVisionTagger(
            {
                first.name: TaggingResult(
                    keyword_tags=["bus"],
                    normalized_tags=["버스"],
                    confidence=0.91,
                    review_status="approved",
                ),
                second.name: TaggingResult(
                    keyword_tags=["apple"],
                    normalized_tags=["사과"],
                    confidence=0.95,
                    review_status="approved",
                ),
            }
        )
    )

    report = service.preview_directory(tmp_path)

    assert report.summary.scanned_count == 2
    assert report.summary.tagged_count == 2
    assert report.summary.reject_count == 0
    assert [result.asset_path for result in report.results] == [
        str(second),
        str(first),
    ]


def test_tag_preview_service_collects_rejects_without_storing_results(tmp_path: Path) -> None:
    valid = tmp_path / "10565_20077_1.svg"
    invalid_name = tmp_path / "wrong.svg"
    invalid_image = tmp_path / "10565_20077_2.png"
    tagging_failure = tmp_path / "10565_20077_3.svg"
    _write_svg(valid, "apple")
    _write_svg(invalid_name, "ignored")
    invalid_image.write_bytes(b"not-a-png")
    _write_svg(tagging_failure, "fail")

    service = TagPreviewService(
        tagger=_ScriptedVisionTagger(
            {
                valid.name: TaggingResult(
                    keyword_tags=["apple"],
                    normalized_tags=["사과"],
                    confidence=0.95,
                    review_status="approved",
                ),
                tagging_failure.name: TaggingResult(
                    keyword_tags=["broken"],
                    normalized_tags=["broken"],
                    confidence=0.10,
                    review_status="pending",
                ),
            },
            failures={tagging_failure.name},
        )
    )

    report = service.preview_directory(tmp_path)

    assert report.summary.scanned_count == 4
    assert report.summary.tagged_count == 1
    assert report.summary.reject_count == 3
    assert report.results[0].asset_path == str(valid)
    assert [(reject.asset_path, reject.reason) for reject in report.summary.rejections] == [
        (str(invalid_image), "invalid_image"),
        (str(tagging_failure), "tagging_failed"),
        (str(invalid_name), "invalid_filename"),
    ]
