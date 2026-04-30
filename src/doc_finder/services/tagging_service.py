from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import base64
import json
import re
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


@dataclass(slots=True)
class RawPreviewResult:
    od_raw: list[str] | None = None
    ocr_raw: str | None = None
    describe_raw: str | None = None


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


def looks_like_object_phrase(chunk: str, stop_prefixes: tuple[str, ...]) -> bool:
    if any(chunk.startswith(prefix) for prefix in stop_prefixes):
        return False
    return len(chunk.split()) <= 3


def normalize_text_chunk(
    chunk: str,
    stop_prefixes: tuple[str, ...],
    stopwords: set[str],
) -> list[str]:
    lowered = " ".join(chunk.lower().split())
    if not lowered:
        return []
    if looks_like_object_phrase(lowered, stop_prefixes):
        return [lowered]
    tokens = [
        token
        for token in re.findall(r"[a-z0-9가-힣]+", lowered)
        if token not in stopwords and len(token) >= 3
    ]
    return tokens[:4]


def compute_confidence(
    primary_signal: bool,
    secondary_signal: bool,
    expansion_signal: bool,
) -> float:
    confidence = 0.45
    if primary_signal:
        confidence += 0.25
    if secondary_signal:
        confidence += 0.15
    if expansion_signal:
        confidence += 0.10
    return round(min(confidence, 0.95), 2)


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
                "content_base64": base64.b64encode(
                    _read_vision_payload_bytes(asset_path)
                ).decode("ascii"),
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

        try:
            # 외부 태거 응답은 배치 전체 안정성에 직접 영향을 주므로 필수 필드는 여기서 검증한다.
            keyword_tags = clean_tag_candidates(
                _require_string_list(data, "keyword_tags")
            )
            normalized_tag_source = clean_tag_candidates(
                _require_string_list(data, "normalized_tags", default=keyword_tags)
            )
            confidence = float(_require_response_field(data, "confidence"))
        except (KeyError, TypeError, ValueError) as exc:
            raise TaggingError(
                f"Vision tagging response is invalid for {asset_path.name}."
            ) from exc

        normalized_tags = build_normalized_tags(
            normalized_tag_source,
            self._query_normalizer,
        )
        return TaggingResult(
            keyword_tags=keyword_tags,
            normalized_tags=normalized_tags,
            confidence=confidence,
            review_status=_resolve_review_status(data),
        )


def _read_vision_payload_bytes(asset_path: Path) -> bytes:
    raw = asset_path.read_bytes()
    if asset_path.suffix.lower() == ".svg":
        raw = _render_svg_to_png_bytes(raw)

    return _flatten_transparent_image_bytes(raw)


def _render_svg_to_png_bytes(raw: bytes) -> bytes:
    try:
        import cairosvg
    except ImportError:
        return raw

    try:
        # Ollama vision 입력은 raster 이미지가 가장 안정적이므로 SVG는 먼저 PNG로 렌더링한다.
        rendered = cairosvg.svg2png(bytestring=raw)
    except Exception:  # noqa: BLE001
        return raw
    if isinstance(rendered, bytes):
        return rendered
    return raw


def _flatten_transparent_image_bytes(raw: bytes) -> bytes:
    try:
        from PIL import Image
    except ImportError:
        return raw

    try:
        with Image.open(BytesIO(raw)) as image:
            if not _image_has_alpha(image):
                return raw

            # 투명 배경을 그대로 보내면 일부 vision 모델이 검정 실루엣으로 오판하므로 흰 배경으로 합성한다.
            source = image.convert("RGBA")
            background = Image.new("RGBA", source.size, (255, 255, 255, 255))
            background.alpha_composite(source)

            output = BytesIO()
            background.convert("RGB").save(output, format="PNG")
            return output.getvalue()
    except OSError:
        return raw


def _image_has_alpha(image) -> bool:
    return "A" in image.getbands() or "transparency" in image.info


def _require_response_field(data: object, field_name: str) -> object:
    if not isinstance(data, dict):
        raise TypeError("Vision tagging response must be a JSON object.")
    return data[field_name]


def _require_string_list(
    data: object,
    field_name: str,
    *,
    default: list[str] | None = None,
) -> list[str]:
    if not isinstance(data, dict):
        raise TypeError("Vision tagging response must be a JSON object.")
    value = data.get(field_name, default)
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list of strings.")
    if not all(isinstance(item, str) for item in value):
        raise TypeError(f"{field_name} must be a list of strings.")
    return value


def _resolve_review_status(data: object) -> str:
    if not isinstance(data, dict):
        return "approved"
    value = data.get("review_status", "approved")
    if not isinstance(value, str) or not value.strip():
        return "approved"
    return value.strip()
