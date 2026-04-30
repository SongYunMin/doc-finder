from pathlib import Path
import json

from doc_finder import cli as cli_module
from doc_finder.schemas.search import TagSearchResponse
from doc_finder.services.ingestion_service import IngestionSummary
from doc_finder.services.tag_preview_service import (
    TagPreviewReport,
    TagPreviewResult,
    TagPreviewSummary,
)


class _StubSearchService:
    def search(self, request):
        return TagSearchResponse.model_validate(
            {
                "results": [
                    {
                        "unit_id": 10565,
                        "data_id": 20077,
                        "image_id": 1,
                        "asset_path": "10565_20077_1.svg",
                        "preview_path": None,
                        "matched_tags": ["apple"],
                        "matched_display_tags": ["apple (사과)"],
                        "confidence": 0.95,
                        "score": 1.0,
                        "cms_ref": {"unit_id": 10565, "data_id": 20077},
                    }
                ]
            }
        )


class _StubIngestionService:
    def ingest_directory(self, path: Path):
        return IngestionSummary(
            scanned_count=1,
            indexed_count=1,
            duplicate_count=0,
            reject_count=0,
        )


class _StubTagPreviewService:
    def preview_directory(self, path: Path):
        del path
        return TagPreviewReport(
            summary=TagPreviewSummary(scanned_count=1, tagged_count=1, reject_count=0),
            results=[
                TagPreviewResult(
                    asset_path="10565_20077_1.svg",
                    od_raw=["apple"],
                    ocr_raw="서윤",
                )
            ],
        )


def test_cli_search_prints_search_result(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        cli_module,
        "build_default_search_service",
        lambda: _StubSearchService(),
    )

    cli_module.main(["search", "--query", "apple"])

    assert "'unit_id': 10565" in capsys.readouterr().out


def test_cli_ingest_prints_summary(capsys, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        cli_module,
        "build_default_ingestion_service",
        lambda progress_reporter=None: _StubIngestionService(),
    )

    cli_module.main(["ingest", "--image-dir", str(tmp_path)])

    assert "'indexed_count': 1" in capsys.readouterr().out


def test_cli_tag_prints_json_preview_result(capsys, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli_module, "build_tagger", lambda provider, **kwargs: object())
    monkeypatch.setattr(
        cli_module,
        "TagPreviewService",
        lambda tagger, **kwargs: _StubTagPreviewService(),
    )

    cli_module.main(["tag", "--image-dir", str(tmp_path)])

    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["tagged_count"] == 1
    assert payload["results"][0]["od_raw"] == ["apple"]
    assert payload["results"][0]["ocr_raw"] == "서윤"


def test_cli_tag_supports_florence2_model_id_override(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_build_tagger(provider, **kwargs):
        captured["provider"] = provider
        captured["environ"] = kwargs["environ"]
        return object()

    monkeypatch.setattr(cli_module, "build_tagger", _fake_build_tagger)
    monkeypatch.setattr(
        cli_module,
        "TagPreviewService",
        lambda tagger, **kwargs: _StubTagPreviewService(),
    )

    cli_module.main(
        [
            "tag",
            "--image-dir",
            str(tmp_path),
            "--tagger-provider",
            "florence2",
            "--florence2-model-id",
            "microsoft/Florence-2-large",
        ]
    )

    assert captured["provider"] == "florence2"
    assert captured["environ"]["DOC_FINDER_TAGGER_PROVIDER"] == "florence2"
    assert captured["environ"]["DOC_FINDER_FLORENCE2_MODEL_ID"] == "microsoft/Florence-2-large"
