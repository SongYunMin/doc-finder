from fastapi import APIRouter, Depends, Request

from doc_finder.schemas.search import TagSearchRequest, TagSearchResponse
from doc_finder.services.search_service import SearchService


router = APIRouter(prefix="/search", tags=["search"])


def get_search_service(request: Request) -> SearchService:
    return request.app.state.search_service


@router.post("/by-tags", response_model=TagSearchResponse)
def search_by_tags(
    request: TagSearchRequest,
    service: SearchService = Depends(get_search_service),
) -> TagSearchResponse:
    return service.search(request)
