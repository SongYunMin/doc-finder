from pathlib import Path

from doc_finder.services.tag_preview_service import TagPreviewService
from doc_finder.services.tagging_service import RawPreviewResult, TaggingError


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
    def __init__(self, mapping: dict[str, RawPreviewResult], failures: set[str] | None = None) -> None:
        self._mapping = mapping
        self._failures = failures or set()

    def preview_raw(self, asset_path: Path, sha256: str) -> RawPreviewResult:
        del sha256
        if asset_path.name in self._failures:
            raise TaggingError(f"failed for {asset_path.name}")
        return self._mapping[asset_path.name]


def test_tag_preview_service_builds_results_for_valid_directory(tmp_path: Path) -> None:
    image = tmp_path / "10565_20077_1.svg"
    _write_svg(image, "apple")

    service = TagPreviewService(
        tagger=_ScriptedVisionTagger(
            {
                image.name: RawPreviewResult(
                    od_raw=["apple"],
                    ocr_raw="서윤",
                )
            }
        )
    )

    report = service.preview_directory(tmp_path)

    assert report.summary.scanned_count == 1
    assert report.summary.tagged_count == 1
    assert report.summary.reject_count == 0
    assert report.results[0].asset_path == str(image)
    assert report.results[0].od_raw == ["apple"]
    assert report.results[0].ocr_raw == "서윤"


def test_tag_preview_service_collects_rejects_without_storing_results(tmp_path: Path) -> None:
    valid = tmp_path / "10565_20077_1.svg"
    invalid_name = tmp_path / "wrong.svg"
    tagging_failure = tmp_path / "10565_20077_2.svg"
    _write_svg(valid, "apple")
    _write_svg(invalid_name, "ignored")
    _write_svg(tagging_failure, "fail")

    service = TagPreviewService(
        tagger=_ScriptedVisionTagger(
            {
                valid.name: RawPreviewResult(od_raw=["apple"], ocr_raw="서윤"),
                tagging_failure.name: RawPreviewResult(od_raw=["broken"]),
            },
            failures={tagging_failure.name},
        )
    )

    report = service.preview_directory(tmp_path)

    assert report.summary.scanned_count == 3
    assert report.summary.tagged_count == 1
    assert report.summary.reject_count == 2
    assert [(reject.asset_path, reject.reason) for reject in report.summary.rejections] == [
        (str(tagging_failure), "tagging_failed"),
        (str(invalid_name), "invalid_filename"),
    ]
