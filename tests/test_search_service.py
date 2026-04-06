from doc_finder.schemas.search import TextSearchRequest, TextSearchResponse
from doc_finder.services.search_service import SearchService


def test_search_service_returns_validated_response() -> None:
    service = SearchService()

    result = service.search(TextSearchRequest(query="service layer"))

    assert result == TextSearchResponse(
        query="service layer",
        answer="service layer",
    )
