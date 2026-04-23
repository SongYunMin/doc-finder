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
            keyword_tags=["apple"],
            normalized_tags=["apple"],
            tag_text_projection="apple 사과",
            confidence=0.95,
            review_status="approved",
            embedding=embedding_service.embed_text("apple 사과"),
            tagged_at=datetime.now(UTC),
        )
    )
    service = SearchService(
        repository=repository,
        query_normalizer=QueryNormalizer(),
        embedding_service=embedding_service,
    )

    result = service.search(TagSearchRequest(query="사과", top_k=3))

    assert result.results[0].unit_id == 10565
    assert result.results[0].data_id == 20077
    assert result.results[0].matched_tags == ["apple"]
    assert result.results[0].matched_display_tags == ["apple (사과)"]
    assert result.results[0].cms_ref.model_dump() == {
        "unit_id": 10565,
        "data_id": 20077,
    }
    assert "question_json" not in result.model_dump()["results"][0]


def test_search_service_matches_text_namespace_tags_from_raw_query() -> None:
    repository = InMemoryImageIndexRepository()
    embedding_service = HashingEmbeddingService()
    repository.upsert_image(
        ImageDocument(
            unit_id=45116,
            data_id=175554,
            image_id=2,
            asset_path="45116_175554_2.svg",
            sha256="seoyun",
            file_size=120,
            width=120,
            height=80,
            keyword_tags=["triangle", "text:서윤"],
            normalized_tags=["triangle", "text:서윤"],
            tag_text_projection="triangle text:서윤 서윤",
            confidence=0.92,
            review_status="approved",
            embedding=embedding_service.embed_text("triangle text:서윤 서윤"),
            tagged_at=datetime.now(UTC),
        )
    )
    service = SearchService(
        repository=repository,
        query_normalizer=QueryNormalizer(),
        embedding_service=embedding_service,
    )

    result = service.search(TagSearchRequest(query="서윤", top_k=3))

    assert result.results[0].matched_tags == ["text:서윤"]
