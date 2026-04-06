# doc-finder

Very small Python LangGraph starter for this workspace.

## What is included

- `pyproject.toml`: minimal dependency and pytest setup
- `src/doc_finder/graph.py`: one-node `StateGraph`
- `src/doc_finder/main.py`: simple CLI entrypoint
- `src/doc_finder/api.py`: FastAPI wrapper over the graph
- `tests/test_graph.py`: smoke test for graph invocation
- `tests/test_api.py`: smoke tests for HTTP endpoints

## Recommended Python version

Use Python `3.13`.

`langchain-core` currently emits a compatibility warning on Python `3.14+`, so this project explicitly targets `<3.14`.

## Install

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Run

```bash
python -m doc_finder.main --query "hello"
```

## Run API

```bash
uvicorn doc_finder.api:app --reload
```

Available endpoints:

- `GET /health`
- `POST /invoke`

Example:

```bash
curl -X POST http://127.0.0.1:8000/invoke \
  -H 'Content-Type: application/json' \
  -d '{"query":"hello"}'
```

## Test

```bash
pytest -q
```
