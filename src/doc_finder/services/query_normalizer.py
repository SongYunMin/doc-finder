from __future__ import annotations

from dataclasses import dataclass


DEFAULT_SYNONYMS = {
    "apple": "사과",
    "apples": "사과",
    "red apple": "사과",
    "bus": "버스",
    "buses": "버스",
}


@dataclass(slots=True)
class NormalizedQuery:
    original_query: str
    cleaned_query: str
    normalized_query: str
    variants: list[str]


class QueryNormalizer:
    def __init__(self, synonyms: dict[str, str] | None = None) -> None:
        self._synonyms = dict(DEFAULT_SYNONYMS)
        if synonyms:
            self._synonyms.update(synonyms)

    def normalize(self, query: str) -> NormalizedQuery:
        cleaned_query = " ".join(query.split()).casefold()
        normalized_query = self._synonyms.get(cleaned_query, cleaned_query)
        variants = sorted({cleaned_query, normalized_query})
        return NormalizedQuery(
            original_query=query,
            cleaned_query=cleaned_query,
            normalized_query=normalized_query,
            variants=variants,
        )

    def normalize_tag(self, tag: str) -> str:
        cleaned_tag = " ".join(tag.split()).casefold()
        return self._synonyms.get(cleaned_tag, cleaned_tag)

    def normalize_tag_candidates(self, tag: str) -> list[str]:
        cleaned_tag = " ".join(tag.split()).casefold()
        direct_match = self._synonyms.get(cleaned_tag)
        if direct_match:
            return [direct_match]

        normalized_tokens = []
        for token in cleaned_tag.replace(",", " ").split():
            mapped = self._synonyms.get(token)
            if mapped:
                normalized_tokens.append(mapped)

        if normalized_tokens:
            return sorted(dict.fromkeys(normalized_tokens))

        return [cleaned_tag]
