from fastapi.testclient import TestClient

from doc_finder.api import app


client = TestClient(app)


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_invoke_endpoint_returns_graph_result() -> None:
    response = client.post("/invoke", json={"query": "hello api"})

    assert response.status_code == 200
    assert response.json() == {
        "query": "hello api",
        "answer": "hello api",
    }
