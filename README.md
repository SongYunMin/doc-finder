# doc-finder

FastAPI 기반 이미지 태그 검색 서버다. 현재 테스트 환경에서는 핵심 경로만 유지한다.

- 이미지 디렉터리를 스캔해 태그 인덱스를 만든다.
- 정규화된 태그와 간단한 텍스트 임베딩으로 검색한다.
- CMS 원본을 복제하지 않고 `unit_id`, `data_id`, `image_id` 참조만 반환한다.

## Structure

- `src/doc_finder/app.py`: FastAPI app factory
- `src/doc_finder/api/`: HTTP routes
- `src/doc_finder/services/`: ingestion, tagging, search, normalization
- `src/doc_finder/repositories/`: in-memory/Postgres index repositories
- `src/doc_finder/graphs/`: LangGraph search workflow
- `src/doc_finder/taggers/`: core tagger registry and providers
- `tests/`: API, ingestion, repository, search smoke tests

## Asset Filename Contract

이미지 파일명은 아래 형식을 따른다.

```text
<unitId>_<dataId>_<imageOrder>.svg
<unitId>_<dataId>_<imageOrder>.png
<unitId>_<dataId>.svg
<unitId>_<dataId>.png
```

- `unitId`: 문항 개념 식별자
- `dataId`: 같은 문항 개념 아래의 콘텐츠 식별자
- `imageOrder`: 같은 `unitId + dataId` 조합 안의 이미지 순서
- `imageOrder`가 없으면 기본값 `1`

## Install

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Environment

`.env.example`을 복사해 `.env`를 만들고 실행 전에 로드한다.

```bash
cp .env.example .env
set -a
source .env
set +a
```

테스트 환경에서는 `static` provider가 가장 단순하다.

```dotenv
DOC_FINDER_DATABASE_URL=postgresql://postgres:postgres@localhost:5431/doc_finder
DOC_FINDER_TAGGER_PROVIDER=static
DOC_FINDER_STATIC_TAGS=/Users/knowre-yunmin/doc-finder/static-tags.json
```

외부 태깅 서비스를 붙일 때는 `http` provider만 사용한다.

```dotenv
DOC_FINDER_TAGGER_PROVIDER=http
DOC_FINDER_VISION_ENDPOINT=https://example.com/vision/tag
DOC_FINDER_VISION_API_KEY=your_api_key
```

## Tagger Providers

- `static`: JSON fixture에 정의된 태그를 사용한다. 로컬 스모크 테스트용이다.
- `http`: 외부 태깅 API에 이미지 base64 payload를 보내고 태그 응답을 받는다.

로컬 Florence/PaliGemma/SearchTag 튜닝 코드는 현재 핵심 경로에서 제거했다. 다시 필요해지면 별도 실험 브랜치나 외부 태깅 서비스로 분리해서 붙이는 편이 안전하다.

## Run API

```bash
python -m doc_finder.main --reload
```

Endpoints:

- `GET /health`
- `POST /search/by-tags`

Example:

```bash
curl -X POST http://127.0.0.1:8000/search/by-tags \
  -H 'Content-Type: application/json' \
  -d '{"query":"사과","top_k":5}'
```

## CLI

```bash
python -m doc_finder.cli ingest --image-dir ./images
python -m doc_finder.cli search --query "apple" --top-k 5
```

주의: `DOC_FINDER_DATABASE_URL`이 없으면 in-memory 저장소를 쓴다. 이 경우 `ingest`와 `search`를 별도 CLI 프로세스로 실행하면 데이터가 이어지지 않는다.

## Test

```bash
.venv/bin/python -m pytest -q
```
