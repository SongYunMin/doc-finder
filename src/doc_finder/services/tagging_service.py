from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import base64
import json
import string
from typing import TYPE_CHECKING, Protocol
import urllib.request

if TYPE_CHECKING:
    from doc_finder.services.query_normalizer import QueryNormalizer


class TaggingError(RuntimeError):
    pass


@dataclass(slots=True)
class TaggingResult:
    keyword_tags: list[str]
    normalized_tags: list[str]
    confidence: float
    review_status: str = "approved"


class VisionTagger(Protocol):
    def tag(self, asset_path: Path, sha256: str) -> TaggingResult:
        """이미지 자산을 검색용 태그 계약으로 변환한다."""


def clean_tag_candidates(candidates: list[str], max_tags: int | None = None) -> list[str]:
    cleaned_candidates: list[str] = []
    seen: set[str] = set()

    for candidate in candidates:
        cleaned = " ".join(candidate.strip().lower().split())
        if not cleaned:
            continue
        if all(char in string.punctuation for char in cleaned):
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        cleaned_candidates.append(cleaned)
        if max_tags is not None and len(cleaned_candidates) >= max_tags:
            break

    return cleaned_candidates


def build_normalized_tags(keyword_tags: list[str], query_normalizer: QueryNormalizer) -> list[str]:
    normalized: list[str] = []
    for tag in keyword_tags:
        normalized.extend(query_normalizer.normalize_tag_candidates(tag))
    return clean_tag_candidates(normalized)


class StaticVisionTagger:
    def __init__(self, mapping: dict[str, TaggingResult]) -> None:
        self._mapping = mapping

    def tag(self, asset_path: Path, sha256: str) -> TaggingResult:
        try:
            return self._mapping[asset_path.name]
        except KeyError as exc:
            raise TaggingError(f"No static tags configured for {asset_path.name}.") from exc


class HttpVisionTagger:
    def __init__(
        self,
        endpoint_url: str,
        query_normalizer: QueryNormalizer,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._endpoint_url = endpoint_url
        self._query_normalizer = query_normalizer
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    def tag(self, asset_path: Path, sha256: str) -> TaggingResult:
        payload = json.dumps(
            {
                "filename": asset_path.name,
                "sha256": sha256,
                "content_base64": base64.b64encode(asset_path.read_bytes()).decode("ascii"),
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        request = urllib.request.Request(
            self._endpoint_url,
            data=payload,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise TaggingError(f"Vision tagging failed for {asset_path.name}.") from exc

        keyword_tags = clean_tag_candidates(list(data["keyword_tags"]))
        normalized_tags = build_normalized_tags(
            clean_tag_candidates(list(data.get("normalized_tags", data["keyword_tags"]))),
            self._query_normalizer,
        )
        return TaggingResult(
            keyword_tags=keyword_tags,
            normalized_tags=normalized_tags,
            confidence=float(data["confidence"]),
            review_status=str(data.get("review_status", "approved")),
        )
