from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class SearchTagTaxonomy:
    task_name: str
    prompt_prefix: str
    canonical_tags: frozenset[str]
    display_aliases_ko: dict[str, tuple[str, ...]]
    query_aliases_ko: dict[str, str]


def _normalize_key(value: str) -> str:
    return " ".join(str(value).split()).casefold()


def _default_taxonomy_path() -> Path:
    override = os.getenv("DOC_FINDER_SEARCH_TAG_TAXONOMY_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return (Path(__file__).resolve().parent / "search_tag_taxonomy.yaml").resolve()


@lru_cache(maxsize=4)
def load_search_tag_taxonomy(path: str | None = None) -> SearchTagTaxonomy:
    taxonomy_path = Path(path).expanduser().resolve() if path else _default_taxonomy_path()
    payload = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8")) or {}

    canonical_tags = frozenset(
        _normalize_key(tag)
        for tag in payload.get("canonical_tags", [])
        if _normalize_key(tag)
    )

    display_aliases_ko: dict[str, tuple[str, ...]] = {}
    for canonical_tag, aliases in dict(payload.get("display_aliases_ko", {})).items():
        normalized_canonical = _normalize_key(canonical_tag)
        if not normalized_canonical:
            continue
        values = tuple(
            dict.fromkeys(
                alias.strip()
                for alias in list(aliases or [])
                if str(alias).strip()
            )
        )
        display_aliases_ko[normalized_canonical] = values

    query_aliases_ko: dict[str, str] = {}
    for alias, canonical_tag in dict(payload.get("query_aliases_ko", {})).items():
        normalized_alias = _normalize_key(alias)
        normalized_canonical = _normalize_key(canonical_tag)
        if not normalized_alias or not normalized_canonical:
            continue
        query_aliases_ko[normalized_alias] = normalized_canonical

    return SearchTagTaxonomy(
        task_name=str(payload.get("task_name", "searchtag")).strip() or "searchtag",
        prompt_prefix=str(payload.get("prompt_prefix", "<SearchTag>")).strip() or "<SearchTag>",
        canonical_tags=canonical_tags,
        display_aliases_ko=display_aliases_ko,
        query_aliases_ko=query_aliases_ko,
    )
