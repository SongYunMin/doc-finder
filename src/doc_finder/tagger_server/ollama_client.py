from __future__ import annotations

import json
import re
import urllib.request

from doc_finder.tagger_server.prompt import TAGGING_PROMPT
from doc_finder.tagger_server.schemas import VisionTagRequest, VisionTagResponse


class OllamaTaggingError(RuntimeError):
    pass


class OllamaVisionTaggerClient:
    def __init__(
        self,
        *,
        ollama_url: str,
        model: str,
        timeout_seconds: float,
    ) -> None:
        self._ollama_url = ollama_url.rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds

    def tag(self, request: VisionTagRequest) -> VisionTagResponse:
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": TAGGING_PROMPT,
                    "images": [request.content_base64],
                }
            ],
            "format": "json",
            "stream": False,
            "think": False,
            "options": {"temperature": 0},
        }
        http_request = urllib.request.Request(
            f"{self._ollama_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                http_request,
                timeout=self._timeout_seconds,
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise OllamaTaggingError("Ollama tagging failed.") from exc

        try:
            content = data["message"]["content"]
            return _normalize_tag_response(json.loads(_strip_json_code_block(content)))
        except Exception as exc:  # noqa: BLE001
            raise OllamaTaggingError("Ollama tagging response is invalid.") from exc


def _strip_json_code_block(content: object) -> str:
    text = str(content).strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _normalize_tag_response(data: object) -> VisionTagResponse:
    if not isinstance(data, dict):
        raise TypeError("Tagging response must be a JSON object.")

    keyword_tags = _clean_string_list(data.get("keyword_tags", []))
    normalized_tags = _clean_string_list(data.get("normalized_tags", keyword_tags))
    confidence = _clamp_confidence(data.get("confidence", 0.0))
    review_status = data.get("review_status", "pending")
    if review_status not in {"approved", "pending"}:
        review_status = "pending"

    return VisionTagResponse(
        keyword_tags=keyword_tags,
        normalized_tags=normalized_tags,
        confidence=confidence,
        review_status=str(review_status),
    )


def _clean_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        raise TypeError("Tag fields must be lists.")

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise TypeError("Tag fields must contain only strings.")
        tag = " ".join(item.strip().split())
        if not tag or tag in seen:
            continue
        seen.add(tag)
        cleaned.append(tag)
    return cleaned


def _clamp_confidence(value: object) -> float:
    confidence = float(value)
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence
