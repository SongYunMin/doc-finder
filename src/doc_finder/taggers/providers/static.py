from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

from doc_finder.services.query_normalizer import QueryNormalizer
from doc_finder.services.tagging_service import (
    StaticVisionTagger,
    TaggingResult,
    build_normalized_tags,
    clean_tag_candidates,
)
from doc_finder.taggers.registry import register_tagger_provider


def build_static_tagger(
    *,
    query_normalizer: QueryNormalizer,
    environ: Mapping[str, str],
):
    mapping_path = environ.get("DOC_FINDER_STATIC_TAGS")
    if not mapping_path:
        raise ValueError(
            "DOC_FINDER_STATIC_TAGS must point to a JSON file for the static tagger."
        )

    payload = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    return StaticVisionTagger(
        {
            filename: TaggingResult(
                keyword_tags=clean_tag_candidates(list(tags["keyword_tags"])),
                normalized_tags=build_normalized_tags(
                    clean_tag_candidates(list(tags.get("normalized_tags", tags["keyword_tags"]))),
                    query_normalizer,
                ),
                confidence=float(tags["confidence"]),
                review_status=str(tags.get("review_status", "approved")),
            )
            for filename, tags in payload.items()
        }
    )


register_tagger_provider("static", build_static_tagger)
