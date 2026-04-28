from fastapi.testclient import TestClient

from doc_finder.app import create_app
from doc_finder.repositories.image_index import InMemoryImageIndexRepository, ImageDocument
from doc_finder.services.embedding_service import HashingEmbeddingService
from doc_finder.services.query_normalizer import QueryNormalizer
from doc_finder.services.search_service import SearchService


def _build_client() -> TestClient:
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
            tagged_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
        )
    )
    service = SearchService(
        repository=repository,
        query_normalizer=QueryNormalizer(),
        embedding_service=embedding_service,
    )
    return TestClient(create_app(search_service=service))


def test_health_endpoint_returns_ok() -> None:
    client = _build_client()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_by_tags_endpoint_returns_tag_results() -> None:
    client = _build_client()
    response = client.post("/search/by-tags", json={"query": "사과", "top_k": 5})

    assert response.status_code == 200
    assert response.json()["results"][0]["unit_id"] == 10565
    assert response.json()["results"][0]["data_id"] == 20077
    assert "question_json" not in response.json()["results"][0]


def test_text_search_endpoint_is_not_part_of_minimal_api() -> None:
    client = _build_client()
    response = client.post("/search/text", json={"query": "apple", "top_k": 5})

    assert response.status_code == 404
