from datetime import UTC, datetime

from doc_finder.repositories.image_index import InMemoryImageIndexRepository, ImageDocument
from doc_finder.schemas.search import TagSearchRequest
from doc_finder.services.embedding_service import HashingEmbeddingService
from doc_finder.services.query_normalizer import QueryNormalizer
from doc_finder.services.search_service import SearchService


def test_search_service_returns_cms_reference_without_question_json() -> None:
    repository = InMemoryImageIndexRepository()
    embedding_service = HashingEmbeddingService()
    repository.upsert_image(
        ImageDocument(
            unit_id=10565,
            data_id=20077,
            image_id=1,
            asset_path="10565_20077_1.svg",
            sha256="apple",
            file_size=120,
            width=120,
            height=80,
            keyword_tags=["사과", "apple"],
            normalized_tags=["사과"],
            tag_text_projection="사과 apple",
            confidence=0.95,
            review_status="approved",
            embedding=embedding_service.embed_text("사과 apple"),
            tagged_at=datetime.now(UTC),
        )
    )
    service = SearchService(
        repository=repository,
        query_normalizer=QueryNormalizer(),
        embedding_service=embedding_service,
    )

    result = service.search(TagSearchRequest(query="apple", top_k=3))

    assert result.results[0].unit_id == 10565
    assert result.results[0].data_id == 20077
    assert result.results[0].cms_ref.model_dump() == {
        "unit_id": 10565,
        "data_id": 20077,
    }
    assert "question_json" not in result.model_dump()["results"][0]
