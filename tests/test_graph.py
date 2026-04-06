from doc_finder.graphs.search_graph import build_text_search_graph


def test_graph_echoes_query_into_answer() -> None:
    graph = build_text_search_graph()

    result = graph.invoke({"query": "langgraph basic setup"})

    assert result == {
        "query": "langgraph basic setup",
        "answer": "langgraph basic setup",
    }
