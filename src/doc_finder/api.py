from fastapi import FastAPI
from pydantic import BaseModel

from doc_finder.graph import build_graph


class InvokeRequest(BaseModel):
    query: str


class InvokeResponse(BaseModel):
    query: str
    answer: str


def create_app() -> FastAPI:
    app = FastAPI(title="doc-finder", version="0.1.0")
    graph = build_graph()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/invoke", response_model=InvokeResponse)
    def invoke(request: InvokeRequest) -> InvokeResponse:
        result = graph.invoke({"query": request.query})
        return InvokeResponse.model_validate(result)

    return app


app = create_app()
