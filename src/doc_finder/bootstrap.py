from __future__ import annotations

from pathlib import Path
import json
import os

from doc_finder.repositories.image_index import (
    InMemoryImageIndexRepository,
    PostgresImageIndexRepository,
)
from doc_finder.services.embedding_service import HashingEmbeddingService
from doc_finder.services.florence2_tagger import Florence2VisionTagger
from doc_finder.services.ingestion_service import IngestionService
from doc_finder.services.query_normalizer import QueryNormalizer
from doc_finder.services.search_service import SearchService
from doc_finder.services.tagging_service import HttpVisionTagger, StaticVisionTagger, TaggingResult


def build_default_repository():
    database_url = os.getenv("DOC_FINDER_DATABASE_URL")
    if database_url:
        # 실제 end-to-end 동작에서는 ingest와 search가 같은 영속 저장소를 봐야 한다.
        return PostgresImageIndexRepository(database_url=database_url)
    return InMemoryImageIndexRepository()


def build_default_search_service() -> SearchService:
    repository = build_default_repository()
    return SearchService(
        repository=repository,
        query_normalizer=QueryNormalizer(),
        embedding_service=HashingEmbeddingService(),
    )


def build_default_ingestion_service() -> IngestionService:
    query_normalizer = QueryNormalizer()
    repository = build_default_repository()
    return IngestionService(
        repository=repository,
        tagger=_build_default_tagger(query_normalizer=query_normalizer),
        embedding_service=HashingEmbeddingService(),
        query_normalizer=query_normalizer,
    )


def _build_default_tagger(query_normalizer: QueryNormalizer | None = None):
    query_normalizer = query_normalizer or QueryNormalizer()
    provider = os.getenv("DOC_FINDER_TAGGER_PROVIDER", "http").casefold()
    if provider == "http":
        # 운영 기본 경로는 외부 비전 태거를 호출하는 HTTP 어댑터다.
        endpoint_url = os.getenv("DOC_FINDER_VISION_ENDPOINT")
        if not endpoint_url:
            raise ValueError(
                "DOC_FINDER_VISION_ENDPOINT must be set when using the http tagger provider."
            )
        return HttpVisionTagger(
            endpoint_url=endpoint_url,
            api_key=os.getenv("DOC_FINDER_VISION_API_KEY"),
        )

    if provider == "static":
        # 실제 태거가 없을 때는 정적 태그 파일로 로컬 스모크 테스트를 할 수 있다.
        mapping_path = os.getenv("DOC_FINDER_STATIC_TAGS")
        if not mapping_path:
            raise ValueError(
                "DOC_FINDER_STATIC_TAGS must point to a JSON file for the static tagger."
            )
        payload = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
        return StaticVisionTagger(
            {
                filename: TaggingResult(
                    keyword_tags=list(tags["keyword_tags"]),
                    normalized_tags=list(tags.get("normalized_tags", tags["keyword_tags"])),
                    confidence=float(tags["confidence"]),
                    review_status=str(tags.get("review_status", "approved")),
                )
                for filename, tags in payload.items()
            }
        )

    if provider == "florence2":
        model_id = os.getenv("DOC_FINDER_FLORENCE2_MODEL_ID", "microsoft/Florence-2-base")
        device = os.getenv("DOC_FINDER_FLORENCE2_DEVICE", _default_florence2_device())
        torch_dtype = os.getenv(
            "DOC_FINDER_FLORENCE2_TORCH_DTYPE",
            "float16" if device == "cuda" else "float32",
        )
        return Florence2VisionTagger(
            model_id=model_id,
            device=device,
            torch_dtype=torch_dtype,
            query_normalizer=query_normalizer,
            max_new_tokens=int(os.getenv("DOC_FINDER_FLORENCE2_MAX_NEW_TOKENS", "512")),
            num_beams=int(os.getenv("DOC_FINDER_FLORENCE2_NUM_BEAMS", "3")),
        )

    raise ValueError(f"Unsupported tagger provider: {provider}")


def _default_florence2_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"
