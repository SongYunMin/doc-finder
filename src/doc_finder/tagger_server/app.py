from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from doc_finder.tagger_server.ollama_client import (
    OllamaTaggingError,
    OllamaVisionTaggerClient,
)
from doc_finder.tagger_server.schemas import VisionTagRequest, VisionTagResponse


def create_app(
    *,
    ollama_url: str | None = None,
    ollama_model: str | None = None,
    timeout_seconds: float | None = None,
) -> FastAPI:
    client = OllamaVisionTaggerClient(
        ollama_url=ollama_url
        or os.getenv("DOC_FINDER_OLLAMA_URL", "http://127.0.0.1:11434"),
        model=ollama_model or os.getenv("DOC_FINDER_OLLAMA_MODEL", "gemma4:31b"),
        timeout_seconds=timeout_seconds
        or float(os.getenv("DOC_FINDER_OLLAMA_TIMEOUT_SECONDS", "120")),
    )
    app = FastAPI(title="doc-finder Gemma tagger", version="0.1.0")
    app.state.ollama_client = client

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/vision/tag", response_model=VisionTagResponse)
    def tag_image(request: VisionTagRequest) -> VisionTagResponse:
        try:
            return app.state.ollama_client.tag(request)
        except OllamaTaggingError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


app = create_app()
