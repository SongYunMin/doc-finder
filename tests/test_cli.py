from pathlib import Path

from doc_finder import cli as cli_module


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
