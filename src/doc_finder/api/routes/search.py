from functools import lru_cache

from fastapi import APIRouter, Depends

from doc_finder.schemas.search import TextSearchRequest, TextSearchResponse
from doc_finder.services.search_service import SearchService


router = APIRouter(prefix="/search", tags=["search"])


@lru_cache
def get_search_service() -> SearchService:
    return SearchService()


@router.post("/text", response_model=TextSearchResponse)
def text_search(
    request: TextSearchRequest,
    service: SearchService = Depends(get_search_service),
) -> TextSearchResponse:
    return service.search(request)
