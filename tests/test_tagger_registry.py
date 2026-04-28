import json
from pathlib import Path

from doc_finder import bootstrap
from doc_finder.services.query_normalizer import QueryNormalizer


def test_build_tagger_from_env_supports_runtime_provider_registration(monkeypatch) -> None:
    from doc_finder.taggers.registry import (
        build_tagger_from_env,
        register_tagger_provider,
    )

    captured: dict[str, object] = {}

    class _FakeTagger:
        def tag(self, asset_path: Path, sha256: str):  # pragma: no cover - protocol shape only
            raise NotImplementedError

    def _fake_builder(*, query_normalizer: QueryNormalizer, environ: dict[str, str]):
        captured["query_normalizer"] = query_normalizer
        captured["provider_env"] = environ["DOC_FINDER_TAGGER_PROVIDER"]
        return _FakeTagger()

    register_tagger_provider("fake-provider", _fake_builder)
    monkeypatch.setenv("DOC_FINDER_TAGGER_PROVIDER", "fake-provider")

    tagger = build_tagger_from_env(query_normalizer=QueryNormalizer())

    assert isinstance(tagger, _FakeTagger)
    assert isinstance(captured["query_normalizer"], QueryNormalizer)
    assert captured["provider_env"] == "fake-provider"


def test_build_default_ingestion_service_uses_generic_tagger_builder(monkeypatch) -> None:
    captured: dict[str, object] = {}
    fake_tagger = object()

    def _fake_builder(*, query_normalizer: QueryNormalizer):
        captured["query_normalizer"] = query_normalizer
        return fake_tagger

    monkeypatch.setattr(bootstrap, "build_tagger_from_env", _fake_builder)

    service = bootstrap.build_default_ingestion_service()

    assert service._tagger is fake_tagger
    assert isinstance(captured["query_normalizer"], QueryNormalizer)


def test_available_tagger_providers_exposes_only_core_providers() -> None:
    from doc_finder.taggers import available_tagger_providers

    providers = available_tagger_providers()

    assert "http" in providers
    assert "static" in providers
    assert "florence2" not in providers
    assert "paligemma2" not in providers


def test_static_tagger_provider_builds_from_json_fixture(tmp_path: Path) -> None:
    from doc_finder.taggers import build_tagger

    mapping_path = tmp_path / "static-tags.json"
    mapping_path.write_text(
        json.dumps(
            {
                "10565_20077_1.png": {
                    "keyword_tags": ["사과", "apple"],
                    "normalized_tags": ["사과"],
                    "confidence": 0.95,
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    tagger = build_tagger(
        "static",
        query_normalizer=QueryNormalizer(),
        environ={"DOC_FINDER_STATIC_TAGS": str(mapping_path)},
    )

    result = tagger.tag(Path("10565_20077_1.png"), "unused")

    assert result.keyword_tags == ["사과", "apple"]
    assert result.normalized_tags == ["apple"]
    assert result.confidence == 0.95
