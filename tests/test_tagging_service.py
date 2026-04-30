import base64
from io import BytesIO
import json
from pathlib import Path

import pytest
from PIL import Image

import doc_finder.services.tagging_service as tagging_module
from doc_finder.services.query_normalizer import QueryNormalizer
from doc_finder.services.tagging_service import HttpVisionTagger, TaggingError


class _FakeHttpResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload, ensure_ascii=False).encode("utf-8")


def test_http_vision_tagger_sends_image_payload_and_normalizes_response(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image = tmp_path / "10565_20077_1.png"
    image.write_bytes(b"fake-image")
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeHttpResponse(
            {
                "keyword_tags": ["사과", "apple"],
                "normalized_tags": ["사과"],
                "confidence": 0.95,
                "review_status": "approved",
            }
        )

    monkeypatch.setattr(tagging_module.urllib.request, "urlopen", fake_urlopen)
    tagger = HttpVisionTagger(
        endpoint_url="https://example.com/vision/tag",
        query_normalizer=QueryNormalizer(),
        api_key="secret",
        timeout_seconds=3.0,
    )

    result = tagger.tag(image, "sha-value")

    assert captured["timeout"] == 3.0
    assert captured["payload"] == {
        "filename": "10565_20077_1.png",
        "sha256": "sha-value",
        "content_base64": base64.b64encode(b"fake-image").decode("ascii"),
    }
    assert result.keyword_tags == ["사과", "apple"]
    assert result.normalized_tags == ["apple"]
    assert result.confidence == 0.95
    assert result.review_status == "approved"


def test_http_vision_tagger_flattens_transparent_png_payload(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image = tmp_path / "10565_20077_1.png"
    # 투명 픽셀의 RGB 값이 검정이어도 모델에는 흰 배경으로 합성된 이미지를 보내야 한다.
    rgba_image = Image.new("RGBA", (2, 1), (0, 0, 0, 0))
    rgba_image.putpixel((1, 0), (255, 0, 0, 255))
    rgba_image.save(image)
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        del timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeHttpResponse(
            {
                "keyword_tags": ["도형"],
                "normalized_tags": ["도형"],
                "confidence": 0.95,
            }
        )

    monkeypatch.setattr(tagging_module.urllib.request, "urlopen", fake_urlopen)
    tagger = HttpVisionTagger(
        endpoint_url="https://example.com/vision/tag",
        query_normalizer=QueryNormalizer(),
    )

    tagger.tag(image, "sha-value")

    payload = captured["payload"]
    assert isinstance(payload, dict)
    flattened_bytes = base64.b64decode(payload["content_base64"])
    flattened_image = Image.open(BytesIO(flattened_bytes))

    assert flattened_image.mode == "RGB"
    assert flattened_image.getpixel((0, 0)) == (255, 255, 255)
    assert flattened_image.getpixel((1, 0)) == (255, 0, 0)


def test_http_vision_tagger_converts_svg_payload_to_white_background_png(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image = tmp_path / "10565_20077_1.svg"
    image.write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg" width="2" height="1">
          <rect x="1" y="0" width="1" height="1" fill="red"/>
        </svg>
        """,
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        del timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeHttpResponse(
            {
                "keyword_tags": ["도형"],
                "normalized_tags": ["도형"],
                "confidence": 0.95,
            }
        )

    monkeypatch.setattr(tagging_module.urllib.request, "urlopen", fake_urlopen)
    tagger = HttpVisionTagger(
        endpoint_url="https://example.com/vision/tag",
        query_normalizer=QueryNormalizer(),
    )

    tagger.tag(image, "sha-value")

    payload = captured["payload"]
    assert isinstance(payload, dict)
    rendered_bytes = base64.b64decode(payload["content_base64"])
    rendered_image = Image.open(BytesIO(rendered_bytes))

    assert rendered_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert rendered_image.mode == "RGB"
    assert rendered_image.getpixel((0, 0)) == (255, 255, 255)
    assert rendered_image.getpixel((1, 0)) == (255, 0, 0)


def test_http_vision_tagger_wraps_invalid_response_shape_as_tagging_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image = tmp_path / "10565_20077_1.png"
    image.write_bytes(b"fake-image")

    def fake_urlopen(request, timeout):
        del request, timeout
        # keyword_tags가 문자열이면 기존 구현은 글자 단위 태그로 잘못 처리할 수 있었다.
        return _FakeHttpResponse({"keyword_tags": "apple", "confidence": 0.95})

    monkeypatch.setattr(tagging_module.urllib.request, "urlopen", fake_urlopen)
    tagger = HttpVisionTagger(
        endpoint_url="https://example.com/vision/tag",
        query_normalizer=QueryNormalizer(),
    )

    with pytest.raises(TaggingError, match="response is invalid"):
        tagger.tag(image, "sha-value")
