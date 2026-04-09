from pathlib import Path

import pytest

from doc_finder import bootstrap
from doc_finder.services.query_normalizer import QueryNormalizer


class _StubFlorenceRunner:
    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses

    def __call__(self, task_prompt: str, asset_path: Path) -> object:
        return self.responses[task_prompt]


def test_build_default_tagger_supports_florence2_provider(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeFlorence2VisionTagger:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("DOC_FINDER_TAGGER_PROVIDER", "florence2")
    monkeypatch.setenv("DOC_FINDER_FLORENCE2_MODEL_ID", "microsoft/Florence-2-base")
    monkeypatch.setenv("DOC_FINDER_FLORENCE2_DEVICE", "cpu")
    monkeypatch.setenv("DOC_FINDER_FLORENCE2_TORCH_DTYPE", "float32")
    monkeypatch.setattr(bootstrap, "Florence2VisionTagger", _FakeFlorence2VisionTagger)

    tagger = bootstrap._build_default_tagger()

    assert isinstance(tagger, _FakeFlorence2VisionTagger)
    assert captured["model_id"] == "microsoft/Florence-2-base"
    assert captured["device"] == "cpu"
    assert captured["torch_dtype"] == "float32"


def test_florence2_tagger_merges_od_and_caption_results_into_keyword_tags() -> None:
    from doc_finder.services.florence2_tagger import Florence2VisionTagger

    tagger = Florence2VisionTagger(
        model_id="microsoft/Florence-2-base",
        device="cpu",
        torch_dtype="float32",
        query_normalizer=QueryNormalizer(),
        prompt_runner=_StubFlorenceRunner(
            {
                "<OD>": {"<OD>": {"labels": ["apple", "apple", "bus"]}},
                "<DETAILED_CAPTION>": {"<DETAILED_CAPTION>": "red apple, school bus, apple"},
            }
        ),
    )

    result = tagger.tag(Path("10565_20077_1.png"), "sha")

    assert result.keyword_tags == ["apple", "bus", "red apple", "school bus"]
    assert result.normalized_tags == ["사과", "버스"]
    assert result.confidence == pytest.approx(0.95)
    assert result.review_status == "approved"


def test_florence2_tagger_marks_low_evidence_result_as_pending() -> None:
    from doc_finder.services.florence2_tagger import Florence2VisionTagger

    tagger = Florence2VisionTagger(
        model_id="microsoft/Florence-2-base",
        device="cpu",
        torch_dtype="float32",
        query_normalizer=QueryNormalizer(),
        prompt_runner=_StubFlorenceRunner(
            {
                "<OD>": {"<OD>": {"labels": []}},
                "<DETAILED_CAPTION>": {"<DETAILED_CAPTION>": "tiny object"},
            }
        ),
    )

    result = tagger.tag(Path("10565_20077_1.png"), "sha")

    assert result.keyword_tags == ["tiny object"]
    assert result.normalized_tags == ["tiny object"]
    assert result.confidence == pytest.approx(0.45)
    assert result.review_status == "pending"
