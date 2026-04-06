from fastapi import APIRouter

from doc_finder.api.routes.health import router as health_router
from doc_finder.api.routes.search import router as search_router


api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(search_router)
