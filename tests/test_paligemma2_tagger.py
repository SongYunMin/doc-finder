from pathlib import Path
import sys

import pytest

from doc_finder import bootstrap
from doc_finder.services.query_normalizer import QueryNormalizer


class _StubPaliGemmaRunner:
    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses

    def __call__(self, task_prompt: str, asset_path: Path) -> object:
        del asset_path
        return self.responses[task_prompt]


def test_paligemma2_tagger_is_available_from_model_namespace() -> None:
    from doc_finder.models.paligemma_2 import PaliGemma2VisionTagger

    assert PaliGemma2VisionTagger.__name__ == "PaliGemma2VisionTagger"
    assert PaliGemma2VisionTagger.__module__ == "doc_finder.models.paligemma_2.tagger"


def test_paligemma2_legacy_service_module_reexports_model_namespace_class() -> None:
    from doc_finder.models.paligemma_2 import PaliGemma2VisionTagger as NamespaceTagger
    from doc_finder.models.paligemma_2 import PaliGemma2VisionTagger as PackageTagger
    from doc_finder.services.paligemma2_tagger import PaliGemma2VisionTagger as LegacyTagger

    assert PackageTagger is NamespaceTagger
    assert LegacyTagger is NamespaceTagger


def test_build_default_tagger_supports_paligemma2_provider(monkeypatch) -> None:
    captured: dict[str, object] = {}
    from doc_finder.taggers.providers import paligemma2 as paligemma2_provider

    class _FakePaliGemma2VisionTagger:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("DOC_FINDER_TAGGER_PROVIDER", "paligemma2")
    monkeypatch.setenv("DOC_FINDER_PALIGEMMA2_MODEL_ID", "google/paligemma2-3b-mix-448")
    monkeypatch.setenv("DOC_FINDER_PALIGEMMA2_DEVICE", "cpu")
    monkeypatch.setenv("DOC_FINDER_PALIGEMMA2_TORCH_DTYPE", "float32")
    monkeypatch.setattr(
        paligemma2_provider,
        "PaliGemma2VisionTagger",
        _FakePaliGemma2VisionTagger,
    )

    tagger = bootstrap._build_default_tagger()

    assert isinstance(tagger, _FakePaliGemma2VisionTagger)
    assert captured["model_id"] == "google/paligemma2-3b-mix-448"
    assert captured["device"] == "cpu"
    assert captured["torch_dtype"] == "float32"


def test_paligemma2_tagger_merges_describe_and_ocr_results_into_keyword_tags() -> None:
    from doc_finder.models.paligemma_2 import PaliGemma2VisionTagger

    tagger = PaliGemma2VisionTagger(
        model_id="google/paligemma2-3b-mix-448",
        device="cpu",
        torch_dtype="float32",
        query_normalizer=QueryNormalizer(),
        prompt_runner=_StubPaliGemmaRunner(
            {
                "describe en": "red apple, school bus",
                "ocr": "A 36",
            }
        ),
    )

    result = tagger.tag(Path("10565_20077_1.png"), "sha")

    assert result.keyword_tags == ["red apple", "school bus", "a", "36"]
    assert result.normalized_tags == ["apple", "bus"]
    assert result.confidence == pytest.approx(0.95)
    assert result.review_status == "approved"


def test_paligemma2_tagger_marks_low_evidence_result_as_pending() -> None:
    from doc_finder.models.paligemma_2 import PaliGemma2VisionTagger

    tagger = PaliGemma2VisionTagger(
        model_id="google/paligemma2-3b-mix-448",
        device="cpu",
        torch_dtype="float32",
        query_normalizer=QueryNormalizer(),
        prompt_runner=_StubPaliGemmaRunner(
            {
                "describe en": "",
                "ocr": "",
            }
        ),
    )

    result = tagger.tag(Path("10565_20077_1.png"), "sha")

    assert result.keyword_tags == []
    assert result.normalized_tags == []
    assert result.confidence == pytest.approx(0.45)
    assert result.review_status == "pending"


def test_paligemma2_tagger_preserves_short_ocr_tokens() -> None:
    from doc_finder.models.paligemma_2 import PaliGemma2VisionTagger

    tagger = PaliGemma2VisionTagger(
        model_id="google/paligemma2-3b-mix-448",
        device="cpu",
        torch_dtype="float32",
        query_normalizer=QueryNormalizer(),
        prompt_runner=_StubPaliGemmaRunner(
            {
                "describe en": "geometry diagram",
                "ocr": "A B C 36 55",
            }
        ),
    )

    result = tagger.tag(Path("10565_20077_1.png"), "sha")

    assert result.keyword_tags == ["geometry diagram", "a", "b", "c", "36", "55"]
    assert result.review_status == "approved"


def test_paligemma2_tagger_preview_raw_returns_describe_and_ocr_outputs() -> None:
    from doc_finder.models.paligemma_2 import PaliGemma2VisionTagger

    tagger = PaliGemma2VisionTagger(
        model_id="google/paligemma2-3b-mix-448",
        device="cpu",
        torch_dtype="float32",
        query_normalizer=QueryNormalizer(),
        prompt_runner=_StubPaliGemmaRunner(
            {
                "describe en": "a geometric diagram with shaded regions",
                "ocr": "서윤",
            }
        ),
    )

    result = tagger.preview_raw(Path("10565_20077_1.png"), "sha")

    assert result.describe_raw == "a geometric diagram with shaded regions"
    assert result.ocr_raw == "서윤"
    assert result.od_raw is None


def test_paligemma2_model_loader_uses_torch_dtype_keyword(monkeypatch) -> None:
    from doc_finder.models.paligemma_2 import PaliGemma2VisionTagger

    captured: dict[str, object] = {}

    class _FakeModel:
        def to(self, device: str):
            captured["to_device"] = device
            return self

    class _FakePaliGemmaForConditionalGeneration:
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

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        type(
            "_FakeTransformers",
            (),
            {
                "PaliGemmaForConditionalGeneration": _FakePaliGemmaForConditionalGeneration,
                "AutoProcessor": _FakeAutoProcessor,
            },
        )(),
    )

    tagger = PaliGemma2VisionTagger(
        model_id="google/paligemma2-3b-mix-448",
        device="cpu",
        torch_dtype="float32",
        query_normalizer=QueryNormalizer(),
        prompt_runner=None,
    )

    tagger._get_model_bundle()

    assert captured["kwargs"]["torch_dtype"] == "float32"
    assert "dtype" not in captured["kwargs"]


def test_run_paligemma_prompt_prefixes_image_token_in_text(tmp_path: Path) -> None:
    import torch
    from PIL import Image

    from doc_finder.models.paligemma_2.tagger import _run_paligemma_prompt

    image_path = tmp_path / "10565_20077_1.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)

    captured: dict[str, object] = {}

    class _FakeProcessor:
        def __call__(self, *, images, text, return_tensors):
            del images, return_tensors
            captured["text"] = text
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "pixel_values": torch.tensor([[[1.0]]]),
            }

        def decode(self, tokens, skip_special_tokens: bool = True):
            del tokens, skip_special_tokens
            return "apple"

    class _FakeModel:
        def generate(self, **kwargs):
            del kwargs
            return torch.tensor([[1, 2, 3, 4]])

    result = _run_paligemma_prompt(
        model=_FakeModel(),
        processor=_FakeProcessor(),
        task_prompt="describe en",
        asset_path=image_path,
        device="cpu",
        torch_dtype="float32",
        max_new_tokens=16,
    )

    assert result == "apple"
    assert captured["text"] == "<image>describe en"


def test_paligemma2_tagger_surfaces_gated_repo_error_with_actionable_hint(monkeypatch) -> None:
    from doc_finder.models.paligemma_2 import PaliGemma2VisionTagger
    from doc_finder.services.tagging_service import TaggingError

    class _FakePaliGemmaForConditionalGeneration:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            del kwargs
            raise OSError(
                "You are trying to access a gated repo. "
                f"Cannot access model {model_id}. Please log in."
            )

    class _FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            del model_id, kwargs
            return object()

    class _FakeTorch:
        float32 = "float32"

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        type(
            "_FakeTransformers",
            (),
            {
                "PaliGemmaForConditionalGeneration": _FakePaliGemmaForConditionalGeneration,
                "AutoProcessor": _FakeAutoProcessor,
            },
        )(),
    )

    tagger = PaliGemma2VisionTagger(
        model_id="google/paligemma2-3b-mix-448",
        device="cpu",
        torch_dtype="float32",
        query_normalizer=QueryNormalizer(),
    )

    with pytest.raises(TaggingError) as exc_info:
        tagger.tag(Path("10565_20077_1.png"), "sha")

    detail = str(exc_info.value)
    assert "Request access" in detail
    assert "HF_TOKEN" in detail
    assert "google/paligemma2-3b-mix-448" in detail
