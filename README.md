# doc-finder

FastAPI-first image tag search server with LangGraph inside the application layer.

## Current structure

- `src/doc_finder/app.py`: FastAPI app entrypoint
- `src/doc_finder/api/`: routers and HTTP layer
- `src/doc_finder/schemas/`: request and response models
- `src/doc_finder/services/`: application service layer
- `src/doc_finder/repositories/`: image index repositories
- `src/doc_finder/graphs/`: LangGraph workflows
- `tests/`: API, service, and graph smoke tests

## Asset filename contract

Image assets must follow this pattern:

```text
<unitId>_<dataId>_<imageOrder>.svg
<unitId>_<dataId>_<imageOrder>.png
```

- `unitId`: 문항 개념 식별자
- `dataId`: 같은 문항 개념 아래의 컨텐츠 식별자
- `imageOrder`: 같은 `unitId + dataId` 조합 안에서의 이미지 순서

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
- `POST /search/by-tags`
- `POST /search/text` (deprecated alias)

Example:

```bash
curl -X POST http://127.0.0.1:8000/search/by-tags \
  -H 'Content-Type: application/json' \
  -d '{"query":"사과","top_k":5}'
```

## Run CLI search

```bash
python -m doc_finder.cli search --query "apple" --top-k 5
```

## Run CLI ingestion

```bash
export DOC_FINDER_DATABASE_URL='postgresql://user:pass@localhost:5431/doc_finder'
export DOC_FINDER_TAGGER_PROVIDER='http'
export DOC_FINDER_VISION_ENDPOINT='https://example.com/vision/tag'

python -m doc_finder.cli ingest --image-dir ./assets
```

## Test

```bash
pytest -q
```
