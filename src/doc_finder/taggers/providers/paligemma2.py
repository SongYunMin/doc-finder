from __future__ import annotations

from collections.abc import Mapping

from doc_finder.models.paligemma_2 import PaliGemma2VisionTagger
from doc_finder.services.query_normalizer import QueryNormalizer
from doc_finder.taggers.providers import _detect_torch_device
from doc_finder.taggers.registry import register_tagger_provider


def build_paligemma2_tagger(
    *,
    query_normalizer: QueryNormalizer,
    environ: Mapping[str, str],
):
    """PaliGemma 2 provider의 환경변수 해석을 한 곳으로 모은다."""

    device = environ.get("DOC_FINDER_PALIGEMMA2_DEVICE", _detect_torch_device())
    torch_dtype = environ.get(
        "DOC_FINDER_PALIGEMMA2_TORCH_DTYPE",
        "float16" if device == "cuda" else "float32",
    )
    return PaliGemma2VisionTagger(
        model_id=environ.get(
            "DOC_FINDER_PALIGEMMA2_MODEL_ID",
            "google/paligemma2-3b-mix-448",
        ),
        device=device,
        torch_dtype=torch_dtype,
        query_normalizer=query_normalizer,
        max_new_tokens=int(environ.get("DOC_FINDER_PALIGEMMA2_MAX_NEW_TOKENS", "128")),
    )


register_tagger_provider("paligemma2", build_paligemma2_tagger)
register_tagger_provider("paligemma_2", build_paligemma2_tagger)
