from __future__ import annotations

from collections.abc import Mapping

from doc_finder.services.query_normalizer import QueryNormalizer
from doc_finder.services.tagging_service import HttpVisionTagger
from doc_finder.taggers.registry import register_tagger_provider


def build_http_tagger(
    *,
    query_normalizer: QueryNormalizer,
    environ: Mapping[str, str],
):
    del query_normalizer

    endpoint_url = environ.get("DOC_FINDER_VISION_ENDPOINT")
    if not endpoint_url:
        raise ValueError(
            "DOC_FINDER_VISION_ENDPOINT must be set when using the http tagger provider."
        )
    return HttpVisionTagger(
        endpoint_url=endpoint_url,
        api_key=environ.get("DOC_FINDER_VISION_API_KEY"),
    )


register_tagger_provider("http", build_http_tagger)
