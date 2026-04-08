from fastapi import FastAPI

from doc_finder.api.router import api_router
from doc_finder.bootstrap import build_default_search_service


def create_app(search_service=None) -> FastAPI:
    app = FastAPI(title="doc-finder", version="0.1.0")
    app.state.search_service = search_service or build_default_search_service()
    app.include_router(api_router)
    return app


app = create_app()
