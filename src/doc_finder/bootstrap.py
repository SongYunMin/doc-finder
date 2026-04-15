from __future__ import annotations

import os

from doc_finder.repositories.image_index import (
    InMemoryImageIndexRepository,
    PostgresImageIndexRepository,
)
from doc_finder.services.embedding_service import HashingEmbeddingService
from doc_finder.services.ingestion_service import IngestionService
from doc_finder.services.query_normalizer import QueryNormalizer
from doc_finder.services.search_service import SearchService
from doc_finder.taggers import build_tagger_from_env


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


def build_default_ingestion_service(progress_reporter=None) -> IngestionService:
    query_normalizer = QueryNormalizer()
    repository = build_default_repository()
    return IngestionService(
        repository=repository,
        tagger=_build_default_tagger(query_normalizer=query_normalizer),
        embedding_service=HashingEmbeddingService(),
        query_normalizer=query_normalizer,
        progress_reporter=progress_reporter,
    )


def _build_default_tagger(query_normalizer: QueryNormalizer | None = None):
    # bootstrap은 provider 구성을 직접 알지 않고 공통 registry에만 위임한다.
    return build_tagger_from_env(query_normalizer=query_normalizer or QueryNormalizer())
