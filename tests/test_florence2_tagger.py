from pathlib import Path
from io import BytesIO
import sys

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


def test_florence2_tagger_reduces_sentence_like_caption_to_object_tokens() -> None:
    from doc_finder.services.florence2_tagger import Florence2VisionTagger

    tagger = Florence2VisionTagger(
        model_id="microsoft/Florence-2-base",
        device="cpu",
        torch_dtype="float32",
        query_normalizer=QueryNormalizer(),
        prompt_runner=_StubFlorenceRunner(
            {
                "<OD>": {"<OD>": {"labels": []}},
                "<DETAILED_CAPTION>": {
                    "<DETAILED_CAPTION>": (
                        "the image shows a person in a wheelchair at night, "
                        "with the grass visible at the bottom."
                    )
                },
            }
        ),
    )

    result = tagger.tag(Path("10565_20077_1.png"), "sha")

    assert result.keyword_tags == ["person", "wheelchair", "grass"]
    assert result.normalized_tags == ["person", "wheelchair", "grass"]
    assert result.review_status == "pending"


def test_florence2_model_loader_uses_torch_dtype_keyword(monkeypatch) -> None:
    from doc_finder.services.florence2_tagger import Florence2VisionTagger

    captured: dict[str, object] = {}

    class _FakeModel:
        def to(self, device: str):
            captured["to_device"] = device
            return self

    class _FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            captured["model_id"] = model_id
            captured["kwargs"] = kwargs
            return _FakeModel()

    class _FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            captured["processor_model_id"] = model_id
            captured["processor_kwargs"] = kwargs
            return object()

    class _FakeTorch:
        float32 = "float32"

    monkeypatch.setitem(__import__("sys").modules, "torch", _FakeTorch())
    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        type(
            "_FakeTransformers",
            (),
            {
                "AutoModelForCausalLM": _FakeAutoModelForCausalLM,
                "AutoProcessor": _FakeAutoProcessor,
            },
        )(),
    )

    tagger = Florence2VisionTagger(
        model_id="microsoft/Florence-2-base",
        device="cpu",
        torch_dtype="float32",
        query_normalizer=QueryNormalizer(),
        prompt_runner=None,
    )

    tagger._get_model_bundle()

    assert captured["kwargs"]["torch_dtype"] == "float32"
    assert "dtype" not in captured["kwargs"]
    assert captured["kwargs"]["attn_implementation"] == "eager"


def test_load_image_as_rgb_supports_svg(monkeypatch, tmp_path: Path) -> None:
    from PIL import Image
    from doc_finder.services.florence2_tagger import _load_image_as_rgb

    svg_path = tmp_path / "10565_20077_1.svg"
    svg_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="10"></svg>',
        encoding="utf-8",
    )

    png_bytes = BytesIO()
    Image.new("RGB", (12, 10), color="white").save(png_bytes, format="PNG")

    class _FakeCairoSvg:
        @staticmethod
        def svg2png(*, bytestring: bytes) -> bytes:
            assert b"<svg" in bytestring
            return png_bytes.getvalue()

    monkeypatch.setitem(sys.modules, "cairosvg", _FakeCairoSvg())

    image = _load_image_as_rgb(svg_path)

    assert image.mode == "RGB"
    assert image.size == (12, 10)
