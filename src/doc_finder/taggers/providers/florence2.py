from __future__ import annotations

from collections.abc import Mapping

from doc_finder.models.florence_2 import Florence2VisionTagger
from doc_finder.services.query_normalizer import QueryNormalizer
from doc_finder.taggers.providers import _detect_torch_device
from doc_finder.taggers.registry import register_tagger_provider


def build_florence2_tagger(
    *,
    query_normalizer: QueryNormalizer,
    environ: Mapping[str, str],
):
    device = environ.get("DOC_FINDER_FLORENCE2_DEVICE", _detect_torch_device())
    torch_dtype = environ.get(
        "DOC_FINDER_FLORENCE2_TORCH_DTYPE",
        "float16" if device == "cuda" else "float32",
    )
    return Florence2VisionTagger(
        model_id=environ.get("DOC_FINDER_FLORENCE2_MODEL_ID", "microsoft/Florence-2-large"),
        device=device,
        torch_dtype=torch_dtype,
        query_normalizer=query_normalizer,
        max_new_tokens=int(environ.get("DOC_FINDER_FLORENCE2_MAX_NEW_TOKENS", "512")),
        num_beams=int(environ.get("DOC_FINDER_FLORENCE2_NUM_BEAMS", "3")),
    )


register_tagger_provider("florence2", build_florence2_tagger)
