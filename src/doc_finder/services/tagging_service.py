from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import base64
import json
import urllib.request


class TaggingError(RuntimeError):
    pass


@dataclass(slots=True)
class TaggingResult:
    keyword_tags: list[str]
    normalized_tags: list[str]
    confidence: float
    review_status: str = "approved"


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
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._endpoint_url = endpoint_url
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

        return TaggingResult(
            keyword_tags=list(data["keyword_tags"]),
            normalized_tags=list(data.get("normalized_tags", data["keyword_tags"])),
            confidence=float(data["confidence"]),
            review_status=str(data.get("review_status", "approved")),
        )
