from pathlib import Path

from doc_finder import bootstrap
from doc_finder.services.query_normalizer import QueryNormalizer


class _StubSearchTagRunner:
    def __init__(self, responses: dict[str, object]) -> None:
        self._responses = responses

    def __call__(self, task_prompt: str, asset_path: Path) -> object:
        del asset_path
        return self._responses[task_prompt]


def test_build_default_tagger_supports_florence2_searchtag_ft_provider(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    from doc_finder.taggers.providers import florence2_searchtag_ft as provider

    class _FakeFlorence2SearchTagVisionTagger:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("DOC_FINDER_TAGGER_PROVIDER", "florence2-searchtag-ft")
    monkeypatch.setenv("DOC_FINDER_FLORENCE2_DEVICE", "cpu")
    monkeypatch.setenv("DOC_FINDER_FLORENCE2_TORCH_DTYPE", "float32")
    monkeypatch.setattr(
        provider,
        "Florence2SearchTagVisionTagger",
        _FakeFlorence2SearchTagVisionTagger,
    )

    tagger = bootstrap._build_default_tagger()

    assert isinstance(tagger, _FakeFlorence2SearchTagVisionTagger)
    assert captured["device"] == "cpu"
    assert captured["torch_dtype"] == "float32"


def test_florence2_searchtag_tagger_parses_semicolon_separated_canonical_tags() -> None:
    from doc_finder.models.florence_2_searchtag_ft import Florence2SearchTagVisionTagger

    tagger = Florence2SearchTagVisionTagger(
        model_id="microsoft/Florence-2-base-ft",
        device="cpu",
        torch_dtype="float32",
        query_normalizer=QueryNormalizer(),
        prompt_runner=_StubSearchTagRunner(
            {
                "<SearchTag>": "apple; rectangle; 직각; 90도",
                "<OCR>": "서윤",
            }
        ),
    )

    result = tagger.tag(Path("10565_20077_1.png"), "sha")

    assert result.keyword_tags == ["apple", "rectangle", "직각", "90도", "text:서윤"]
    assert result.normalized_tags == ["apple", "rectangle", "right_angle", "text:서윤"]
    assert result.review_status == "approved"
