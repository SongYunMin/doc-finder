from fastapi import FastAPI

from doc_finder.api.router import api_router


def create_app() -> FastAPI:
    app = FastAPI(title="doc-finder", version="0.1.0")
    app.include_router(api_router)
    return app


app = create_app()
