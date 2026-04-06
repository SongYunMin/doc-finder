from doc_finder.graphs.search_graph import build_text_search_graph
from doc_finder.schemas.search import TextSearchRequest, TextSearchResponse


class SearchService:
    def __init__(self) -> None:
        self._graph = build_text_search_graph()

    def search(self, request: TextSearchRequest) -> TextSearchResponse:
        result = self._graph.invoke({"query": request.query})
        return TextSearchResponse.model_validate(result)
