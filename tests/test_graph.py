from doc_finder.graph import build_graph


def test_graph_echoes_query_into_answer() -> None:
    graph = build_graph()

    result = graph.invoke({"query": "langgraph basic setup"})

    assert result == {
        "query": "langgraph basic setup",
        "answer": "langgraph basic setup",
    }
