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
