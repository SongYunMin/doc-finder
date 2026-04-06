# doc-finder

FastAPI-first text search server with LangGraph inside the application layer.

## Current structure

- `src/doc_finder/app.py`: FastAPI app entrypoint
- `src/doc_finder/api/`: routers and HTTP layer
- `src/doc_finder/schemas/`: request and response models
- `src/doc_finder/services/`: application service layer
- `src/doc_finder/graphs/`: LangGraph workflows
- `tests/`: API, service, and graph smoke tests

## Recommended Python version

Use Python `3.13`.

## Install

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Run API

```bash
python -m doc_finder.main --reload
```

Available endpoints:

- `GET /health`
- `POST /search/text`

Example:

```bash
curl -X POST http://127.0.0.1:8000/search/text \
  -H 'Content-Type: application/json' \
  -d '{"query":"hello"}'
```

## Run CLI smoke check

```bash
python -m doc_finder.cli --query "hello"
```

## Test

```bash
pytest -q
```
