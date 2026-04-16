import json
from pathlib import Path

from doc_finder import cli as cli_module
from doc_finder.services.tag_preview_service import (
    TagPreviewReport,
    TagPreviewResult,
    TagPreviewSummary,
)


class _StubSearchService:
    def search(self, request):
        return cli_module.TagSearchResponse.model_validate(
            {
                "results": [
                    {
                        "unit_id": 10565,
                        "data_id": 20077,
                        "image_id": 1,
                        "asset_path": "10565_20077_1.svg",
                        "preview_path": None,
                        "matched_tags": ["사과"],
                        "confidence": 0.95,
                        "score": 1.0,
                        "cms_ref": {"unit_id": 10565, "data_id": 20077},
                    }
                ]
            }
        )


class _StubIngestionService:
    def ingest_directory(self, path: Path):
        return cli_module.IngestionSummary(
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
                    keyword_tags=["apple"],
                    normalized_tags=["사과"],
                    confidence=0.95,
                    review_status="approved",
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
    monkeypatch.setattr(
        cli_module,
        "build_tagger",
        lambda provider, **kwargs: object(),
    )
    monkeypatch.setattr(
        cli_module,
        "TagPreviewService",
        lambda tagger: _StubTagPreviewService(),
    )

    cli_module.main(["tag", "--image-dir", str(tmp_path)])

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["summary"]["tagged_count"] == 1
    assert payload["results"][0]["normalized_tags"] == ["사과"]
    assert output.startswith("{\n")
    assert '\n  "summary": {\n' in output
    assert '\n  "results": [\n' in output


def test_cli_tag_prefers_model_id_flag_over_env(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_build_tagger(provider, **kwargs):
        captured["provider"] = provider
        captured["environ"] = kwargs["environ"]
        return object()

    monkeypatch.setenv("DOC_FINDER_FLORENCE2_MODEL_ID", "microsoft/Florence-2-large")
    monkeypatch.setattr(cli_module, "build_tagger", _fake_build_tagger)
    monkeypatch.setattr(
        cli_module,
        "TagPreviewService",
        lambda tagger: _StubTagPreviewService(),
    )

    cli_module.main(
        [
            "tag",
            "--image-dir",
            str(tmp_path),
            "--tagger-provider",
            "florence2",
            "--florence2-model-id",
            "microsoft/Florence-2-large-ft",
        ]
    )

    environ = captured["environ"]
    assert captured["provider"] == "florence2"
    assert environ["DOC_FINDER_TAGGER_PROVIDER"] == "florence2"
    assert environ["DOC_FINDER_FLORENCE2_MODEL_ID"] == "microsoft/Florence-2-large-ft"


def test_cli_tag_supports_florence2_large_ft_provider(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_build_tagger(provider, **kwargs):
        captured["provider"] = provider
        captured["environ"] = kwargs["environ"]
        return object()

    monkeypatch.setattr(cli_module, "build_tagger", _fake_build_tagger)
    monkeypatch.setattr(
        cli_module,
        "TagPreviewService",
        lambda tagger: _StubTagPreviewService(),
    )

    cli_module.main(
        [
            "tag",
            "--image-dir",
            str(tmp_path),
            "--tagger-provider",
            "florence2-large-ft",
        ]
    )

    environ = captured["environ"]
    assert captured["provider"] == "florence2-large-ft"
    assert environ["DOC_FINDER_TAGGER_PROVIDER"] == "florence2-large-ft"


def test_cli_tag_prefers_paligemma2_model_id_flag_over_env(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_build_tagger(provider, **kwargs):
        captured["provider"] = provider
        captured["environ"] = kwargs["environ"]
        return object()

    monkeypatch.setenv("DOC_FINDER_PALIGEMMA2_MODEL_ID", "google/paligemma2-3b-mix-224")
    monkeypatch.setattr(cli_module, "build_tagger", _fake_build_tagger)
    monkeypatch.setattr(
        cli_module,
        "TagPreviewService",
        lambda tagger: _StubTagPreviewService(),
    )

    cli_module.main(
        [
            "tag",
            "--image-dir",
            str(tmp_path),
            "--tagger-provider",
            "paligemma2",
            "--paligemma2-model-id",
            "google/paligemma2-3b-mix-448",
        ]
    )

    environ = captured["environ"]
    assert captured["provider"] == "paligemma2"
    assert environ["DOC_FINDER_TAGGER_PROVIDER"] == "paligemma2"
    assert environ["DOC_FINDER_PALIGEMMA2_MODEL_ID"] == "google/paligemma2-3b-mix-448"
