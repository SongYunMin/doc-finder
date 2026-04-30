import json
import urllib.error

from fastapi.testclient import TestClient

from doc_finder.tagger_server import ollama_client
from doc_finder.tagger_server.app import create_app


class _FakeHttpResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload, ensure_ascii=False).encode("utf-8")


def test_tagger_server_health_returns_ok() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_tagger_server_calls_ollama_and_returns_tag_contract(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeHttpResponse(
            {
                "message": {
                    "content": json.dumps(
                        {
                            "keyword_tags": ["geometry", "rectangular prism"],
                            "normalized_tags": ["직육면체", "입체도형"],
                            "confidence": 0.95,
                            "review_status": "approved",
                        },
                        ensure_ascii=False,
                    )
                }
            }
        )

    monkeypatch.setattr(ollama_client.urllib.request, "urlopen", fake_urlopen)
    client = TestClient(
        create_app(
            ollama_url="http://ollama.local:11434",
            ollama_model="gemma4:31b",
            timeout_seconds=7.0,
        )
    )

    response = client.post(
        "/vision/tag",
        json={
            "filename": "5001_1069.png",
            "sha256": "sha-value",
            "content_base64": "aW1hZ2U=",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "keyword_tags": ["geometry", "rectangular prism"],
        "normalized_tags": ["직육면체", "입체도형"],
        "confidence": 0.95,
        "review_status": "approved",
    }
    assert captured["url"] == "http://ollama.local:11434/api/chat"
    assert captured["timeout"] == 7.0
    payload = captured["payload"]
    assert payload["model"] == "gemma4:31b"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["format"] == "json"
    assert payload["messages"][0]["images"] == ["aW1hZ2U="]


def test_tagger_server_parses_markdown_json_and_normalizes_fields(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        del request, timeout
        return _FakeHttpResponse(
            {
                "message": {
                    "content": """```json
{
  "keyword_tags": [" cube ", "cube", ""],
  "normalized_tags": ["정육면체"],
  "confidence": 1.4,
  "review_status": "complete"
}
```"""
                }
            }
        )

    monkeypatch.setattr(ollama_client.urllib.request, "urlopen", fake_urlopen)
    client = TestClient(create_app())

    response = client.post(
        "/vision/tag",
        json={
            "filename": "cube.png",
            "sha256": "sha-value",
            "content_base64": "aW1hZ2U=",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "keyword_tags": ["cube"],
        "normalized_tags": ["정육면체"],
        "confidence": 1.0,
        "review_status": "pending",
    }


def test_tagger_server_returns_502_for_ollama_failure(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        del request, timeout
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(ollama_client.urllib.request, "urlopen", fake_urlopen)
    client = TestClient(create_app())

    response = client.post(
        "/vision/tag",
        json={
            "filename": "cube.png",
            "sha256": "sha-value",
            "content_base64": "aW1hZ2U=",
        },
    )

    assert response.status_code == 502
    assert "Ollama tagging failed" in response.json()["detail"]
