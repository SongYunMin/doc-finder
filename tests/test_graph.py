from datetime import UTC, datetime

from doc_finder.graphs.search_graph import build_tag_search_graph
from doc_finder.repositories.image_index import InMemoryImageIndexRepository, ImageDocument
from doc_finder.services.embedding_service import HashingEmbeddingService
from doc_finder.services.query_normalizer import QueryNormalizer


def test_graph_prioritizes_exact_tag_hits_over_semantic_candidates() -> None:
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
            keyword_tags=["사과", "apple", "빨간 과일"],
            normalized_tags=["사과"],
            tag_text_projection="사과 apple 빨간 과일",
            confidence=0.96,
            review_status="approved",
            embedding=embedding_service.embed_text("사과 apple 빨간 과일"),
            tagged_at=datetime.now(UTC),
        )
    )
    repository.upsert_image(
        ImageDocument(
            unit_id=10565,
            data_id=20078,
            image_id=1,
            asset_path="10565_20078_1.svg",
            sha256="fruit",
            file_size=120,
            width=120,
            height=80,
            keyword_tags=["과일", "빨간 과일", "apple-like"],
            normalized_tags=["과일"],
            tag_text_projection="과일 빨간 과일 apple-like",
            confidence=0.80,
            review_status="approved",
            embedding=embedding_service.embed_text("과일 빨간 과일 apple-like"),
            tagged_at=datetime.now(UTC),
        )
    )

    graph = build_tag_search_graph(
        repository=repository,
        query_normalizer=QueryNormalizer(),
        embedding_service=embedding_service,
    )

    result = graph.invoke(
        {
            "query": "apple",
            "top_k": 5,
            "review_status": "approved",
        }
    )

    assert result["results"][0]["unit_id"] == 10565
    assert result["results"][0]["data_id"] == 20077
    assert result["results"][0]["matched_tags"] == ["사과"]
    assert result["results"][0]["score"] > result["results"][1]["score"]
