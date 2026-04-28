# DocFinder 프로젝트 통합 요약

작성일: 2026-04-28

## 1. 현재 목적

DocFinder는 이미지 파일을 태그 인덱스로 만들고, 그 인덱스에서 객체명/개념명 검색을 제공하는 FastAPI 서버다.

현재 테스트 환경에서는 핵심 기능만 남긴다.

- 이미지 ingest
- static/http 태거 provider
- 태그 정규화
- in-memory 또는 Postgres 저장
- LangGraph 기반 검색
- FastAPI 검색 API
- CLI ingest/search

CMS는 원본 데이터의 source of truth이고, DocFinder는 검색용 projection만 관리한다.

## 2. 유지한 핵심 구조

- `src/doc_finder/app.py`: FastAPI app factory
- `src/doc_finder/api/`: `/health`, `/search/by-tags`
- `src/doc_finder/services/ingestion_service.py`: 이미지 검증, 태깅, 저장 흐름
- `src/doc_finder/services/tagging_service.py`: 공통 태거 계약, static/http 태거
- `src/doc_finder/services/query_normalizer.py`: 한국어 별칭과 canonical tag 정규화
- `src/doc_finder/repositories/image_index.py`: in-memory/Postgres 저장소
- `src/doc_finder/graphs/search_graph.py`: exact tag + semantic 후보 병합 검색
- `src/doc_finder/taggers/`: provider registry

## 3. Core에서 분리한 범위

프로젝트를 테스트 환경 기준으로 줄이기 위해 core 검색 서버와 직접 관련 없는 모델별 실험 코드를 분리했다.

- FastAPI 앱 내부에서 직접 실행하던 Florence-2 런타임 태거
- Florence-2 large fine-tuned wrapper
- Florence-2 SearchTag fine-tuned wrapper
- PaliGemma 2 런타임 태거
- 태그 raw preview CLI/service
- `models/` 아래 튜닝 문서와 학습 파이프라인 코드
- core 서버 의존성에 섞여 있던 Torch/Transformers/PEFT 계열 의존성

Florence-2를 사용하지 않는다는 뜻은 아니다. 다만 DocFinder core는 모델 실행 환경을 직접 품지 않고, Florence-2는 별도 HTTP 태거 서비스나 실험 브랜치에서 검증한 뒤 `http` provider 뒤에 붙이는 편이 안전하다.

## 4. 데이터 계약

이미지 파일명은 아래 형식을 따른다.

```text
<unitId>_<dataId>_<imageOrder>.svg
<unitId>_<dataId>_<imageOrder>.png
<unitId>_<dataId>.svg
<unitId>_<dataId>.png
```

- `unitId`: 문항 개념 식별자
- `dataId`: 같은 문항 개념 아래의 콘텐츠 식별자
- `imageOrder`: 같은 `unitId + dataId` 안의 이미지 순서
- `imageOrder`가 없으면 기본값 `1`

저장 키는 `unit_id + data_id + image_id`다.

## 5. Ingestion 흐름

1. 이미지 디렉터리를 순회한다.
2. 파일명을 파싱한다.
3. PNG/SVG 메타데이터와 파일 크기를 검증한다.
4. SHA-256 중복을 검사한다.
5. 태거 provider에서 `keyword_tags`, `normalized_tags`, `confidence`를 받는다.
6. `QueryNormalizer`로 검색용 canonical tag를 만든다.
7. tag projection text를 만들고 hashing embedding을 계산한다.
8. 저장소에 upsert한다.

`DOC_FINDER_DATABASE_URL`이 없으면 in-memory 저장소를 쓴다. CLI에서 `ingest`와 `search`를 별도 프로세스로 실행하면 in-memory 데이터는 이어지지 않는다.

## 6. 검색 흐름

현재 검색 API는 하나만 유지한다.

```text
POST /search/by-tags
```

검색은 다음 순서로 동작한다.

1. 질의를 정규화한다.
2. normalized tag exact 후보를 찾는다.
3. tag projection embedding 후보를 찾는다.
4. exact 후보를 semantic 후보보다 우선한다.
5. `unit_id`, `data_id`, `image_id`, `asset_path`, `matched_tags`, `score`, `cms_ref`를 반환한다.

`/search/text` alias는 제거했다. 지금은 검색 표면적을 줄이는 편이 낫다.

## 7. Provider 정책

현재 core provider는 두 개다.

- `static`: 로컬 테스트용 JSON fixture 태거
- `http`: 외부 태깅 API 연동용 태거

로컬 모델 provider는 core 경로에서 분리했다. 대량 태깅을 다시 시도할 때도 앱 내부에 모델 코드를 직접 넣기보다, Florence-2 같은 모델 런타임은 HTTP 태거 서비스로 분리하는 것이 운영상 더 단순하다.

## 8. 남은 리스크

- 현재 embedding은 `HashingEmbeddingService`라 검색 품질 검증용 수준이다.
- semantic fallback은 exact match가 없어도 후보를 반환할 수 있다.
- taxonomy가 얕으면 LLM/태거를 바꿔도 검색 품질이 크게 좋아지지 않는다.
- 30만 건 태깅은 중단 재개, 캐시, 실패 로그, batch checkpoint 없이 바로 돌리면 위험하다.

## 9. 다음 액션

1. `static` provider로 작은 샘플 ingest/search를 안정화한다.
2. Florence-2/외부 LLM/Ollama/Gemma 태깅은 앱 내부 모델 코드가 아니라 `http` provider 뒤에 붙인다.
3. 100~300장 샘플에 대해 expected tag와 expected search query를 먼저 만든다.
4. 필요할 때만 embedding/vector store를 교체한다.
5. 대량 처리 전에 실패 재처리와 진행률 저장 방식을 설계한다.
