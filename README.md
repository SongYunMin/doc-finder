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
<unitId>_<dataId>.svg
<unitId>_<dataId>.png
```

- `unitId`: 문항 개념 식별자
- `dataId`: 같은 문항 개념 아래의 컨텐츠 식별자
- `imageOrder`: 같은 `unitId + dataId` 조합 안에서의 이미지 순서
- `imageOrder`가 없으면 기본값 `1`로 처리한다

## Recommended Python version

Use Python `3.13`.

## Install

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Florence-2를 로컬에서 쓸 때는 첫 실행 시 Hugging Face 모델 다운로드가 발생할 수 있다. CPU에서도 동작은 가능하지만 속도는 느릴 수 있고, 대량 태깅은 GPU가 훨씬 낫다.

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

## Tagger providers

- `static`: 정적 JSON 파일에 적어둔 태그를 그대로 사용한다. 로컬 스모크 테스트에 적합하다.
- `http`: 외부 비전 API가 태그를 생성해 반환한다.
- `florence2`: 로컬 Florence-2 모델을 직접 로드해 태그를 생성한다.

## Environment setup

`.env.example`을 복사해 `.env`를 만들고, 실행 전에 로드한다.

```bash
cp .env.example .env
set -a
source .env
set +a
```

### Florence-2 example

```dotenv
DOC_FINDER_DATABASE_URL=postgresql://postgres:postgres@localhost:5431/doc_finder
DOC_FINDER_TAGGER_PROVIDER=florence2
DOC_FINDER_FLORENCE2_MODEL_ID=microsoft/Florence-2-large
DOC_FINDER_FLORENCE2_DEVICE=cpu
DOC_FINDER_FLORENCE2_TORCH_DTYPE=float32
DOC_FINDER_FLORENCE2_MAX_NEW_TOKENS=512
DOC_FINDER_FLORENCE2_NUM_BEAMS=3
```

### Static example

```dotenv
DOC_FINDER_DATABASE_URL=postgresql://postgres:postgres@localhost:5431/doc_finder
DOC_FINDER_TAGGER_PROVIDER=static
DOC_FINDER_STATIC_TAGS=/Users/knowre-yunmin/doc-finder/static-tags.json
```

## Run CLI search

```bash
python -m doc_finder.cli search --query "apple" --top-k 5
```

## Run CLI ingestion

```bash
python -m doc_finder.cli ingest --image-dir ./images
```

Florence-2 로컬 태거 예시:

```bash
set -a
source .env
set +a

python -m doc_finder.cli ingest --image-dir /absolute/path/to/images
python -m doc_finder.cli search --query "사과" --top-k 5
```

HTTP 태거 예시:

```dotenv
DOC_FINDER_DATABASE_URL=postgresql://postgres:postgres@localhost:5431/doc_finder
DOC_FINDER_TAGGER_PROVIDER=http
DOC_FINDER_VISION_ENDPOINT=https://example.com/vision/tag
DOC_FINDER_VISION_API_KEY=your_api_key
```

주의:
- `DOC_FINDER_DATABASE_URL`이 없으면 메모리 저장소로 떨어진다.
- 이 경우 `ingest`와 `search`가 서로 다른 프로세스라 결과가 이어지지 않는다.
- 실제 end-to-end 확인은 반드시 같은 Postgres를 봐야 한다.

## Test

```bash
pytest -q
```
