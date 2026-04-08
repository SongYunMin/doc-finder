from doc_finder.graphs.search_graph import build_tag_search_graph
from doc_finder.repositories.image_index import InMemoryImageIndexRepository
from doc_finder.schemas.search import TagSearchRequest, TagSearchResponse
from doc_finder.services.embedding_service import HashingEmbeddingService
from doc_finder.services.query_normalizer import QueryNormalizer


class SearchService:
    def __init__(
        self,
        repository=None,
        query_normalizer: QueryNormalizer | None = None,
        embedding_service: HashingEmbeddingService | None = None,
    ) -> None:
        self._repository = repository or InMemoryImageIndexRepository()
        self._query_normalizer = query_normalizer or QueryNormalizer()
        self._embedding_service = embedding_service or HashingEmbeddingService()
        self._repository.ensure_schema()
        self._graph = build_tag_search_graph(
            repository=self._repository,
            query_normalizer=self._query_normalizer,
            embedding_service=self._embedding_service,
        )

    def search(self, request: TagSearchRequest) -> TagSearchResponse:
        result = self._graph.invoke(
            {
                "query": request.query,
                "top_k": request.top_k,
                "review_status": request.review_status,
            }
        )
        return TagSearchResponse.model_validate({"results": result.get("results", [])})
