from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from doc_finder.services.search_tag_taxonomy import load_search_tag_taxonomy


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split()).casefold()


@dataclass(slots=True)
class NormalizedQuery:
    original_query: str
    cleaned_query: str
    normalized_query: str
    variants: list[str]


class QueryNormalizer:
    def __init__(
        self,
        synonyms: dict[str, str] | None = None,
        *,
        taxonomy_path: str | Path | None = None,
    ) -> None:
        self._taxonomy = load_search_tag_taxonomy(
            str(Path(taxonomy_path).expanduser().resolve()) if taxonomy_path else None
        )
        self._canonical_tags = set(self._taxonomy.canonical_tags)
        self._alias_to_canonical = {
            alias: canonical
            for alias, canonical in self._taxonomy.query_aliases_ko.items()
        }
        for canonical in self._canonical_tags:
            self._alias_to_canonical.setdefault(canonical, canonical)
        if synonyms:
            self._alias_to_canonical.update(
                {
                    _normalize_whitespace(alias): _normalize_whitespace(canonical)
                    for alias, canonical in synonyms.items()
                    if _normalize_whitespace(alias) and _normalize_whitespace(canonical)
                }
            )

    def normalize(self, query: str) -> NormalizedQuery:
        cleaned_query = _normalize_whitespace(query)
        normalized_candidates = self.normalize_tag_candidates(cleaned_query)
        normalized_query = normalized_candidates[0] if normalized_candidates else cleaned_query
        text_variant = self._build_text_tag(cleaned_query)
        variants = list(
            dict.fromkeys(
                tag
                for tag in [cleaned_query, normalized_query, text_variant]
                if tag
            )
        )
        return NormalizedQuery(
            original_query=query,
            cleaned_query=cleaned_query,
            normalized_query=normalized_query,
            variants=variants,
        )

    def normalize_tag(self, tag: str) -> str:
        candidates = self.normalize_tag_candidates(tag)
        return candidates[0] if candidates else ""

    def normalize_tag_candidates(self, tag: str) -> list[str]:
        cleaned_tag = _normalize_whitespace(tag)
        if not cleaned_tag:
            return []

        if self._is_text_tag(cleaned_tag):
            return [cleaned_tag]

        direct_match = self._alias_to_canonical.get(cleaned_tag)
        if direct_match:
            return [direct_match]

        normalized_tokens = []
        for token in cleaned_tag.replace(",", " ").split():
            mapped = self._alias_to_canonical.get(token)
            if mapped:
                normalized_tokens.append(mapped)

        if normalized_tokens:
            return list(dict.fromkeys(normalized_tokens))

        return []

    def display_tag(self, tag: str) -> str:
        cleaned_tag = _normalize_whitespace(tag)
        if self._is_text_tag(cleaned_tag):
            return cleaned_tag
        canonical_tag = cleaned_tag if cleaned_tag in self._canonical_tags else self._alias_to_canonical.get(cleaned_tag)
        if canonical_tag is None:
            return cleaned_tag

        display_aliases = self._taxonomy.display_aliases_ko.get(canonical_tag, ())
        if not display_aliases:
            return canonical_tag
        return f"{canonical_tag} ({', '.join(display_aliases)})"

    def display_tags(self, tags: list[str]) -> list[str]:
        rendered: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            display_tag = self.display_tag(tag)
            if not display_tag or display_tag in seen:
                continue
            seen.add(display_tag)
            rendered.append(display_tag)
        return rendered

    def projection_terms(self, canonical_tags: list[str]) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()
        for canonical_tag in canonical_tags:
            normalized_canonical = self.normalize_tag(canonical_tag)
            if not normalized_canonical:
                continue
            if self._is_text_tag(normalized_canonical):
                raw_text = normalized_canonical.split(":", 1)[1].strip()
                for term in (normalized_canonical, raw_text):
                    if not term or term in seen:
                        continue
                    seen.add(term)
                    terms.append(term)
                continue
            for term in (
                normalized_canonical,
                *self._taxonomy.display_aliases_ko.get(normalized_canonical, ()),
            ):
                cleaned_term = str(term).strip()
                if not cleaned_term or cleaned_term in seen:
                    continue
                seen.add(cleaned_term)
                terms.append(cleaned_term)
        return terms

    def _build_text_tag(self, value: str) -> str | None:
        cleaned_value = _normalize_whitespace(value)
        if not cleaned_value:
            return None
        if cleaned_value in self._alias_to_canonical:
            return None
        if self._is_text_tag(cleaned_value):
            return cleaned_value
        return f"text:{cleaned_value}"

    @staticmethod
    def _is_text_tag(value: str) -> bool:
        return value.startswith("text:") and len(value.split(":", 1)[1].strip()) > 0
