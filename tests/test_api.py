from fastapi.testclient import TestClient

from doc_finder.app import app


client = TestClient(app)


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_text_search_endpoint_returns_graph_result() -> None:
    response = client.post("/search/text", json={"query": "hello api"})

    assert response.status_code == 200
    assert response.json() == {
        "query": "hello api",
        "answer": "hello api",
    }
