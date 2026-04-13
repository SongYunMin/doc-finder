# Florence-2 로컬 태거 구현 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 외부 비전 API 없이도 이미지 태그를 생성할 수 있도록 로컬 Florence-2 provider를 추가한다.

**Architecture:** 기존 `TaggingResult` 계약과 ingestion 흐름은 유지한다. 새 `florence2` 태거 provider를 추가해 로컬 Hugging Face Florence-2 모델을 한 번만 로드하고, 이미지마다 `<OD>`와 `<DETAILED_CAPTION>` 프롬프트를 실행한 뒤 결과를 병합해 `keyword_tags`를 만든다. 이 단계에서는 검색/인덱스 스키마는 바꾸지 않는다.

**Tech Stack:** Python 3.13, FastAPI, LangGraph, Hugging Face Transformers, Torch, Pillow, pytest

---

### Task 1: provider 계약을 실패 테스트로 먼저 고정

**Files:**
- Create: `tests/test_florence2_tagger.py`
- Modify: `tests/test_ingestion_service.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_search_service.py`

**Step 1: 실패 테스트 작성**

다음 요구사항을 강제하는 테스트를 추가한다.
- `Florence2VisionTagger`가 기존 태거와 같은 `.tag(asset_path, sha256)` 계약을 제공해야 한다.
- `bootstrap._build_default_tagger()`가 `DOC_FINDER_TAGGER_PROVIDER=florence2`를 받아들여야 한다.
- Florence-2의 `<OD>`와 `<DETAILED_CAPTION>` 출력이 안정적인 `keyword_tags`로 병합되어야 한다.
- 중복 태그와 빈 문자열은 인덱싱 전에 제거되어야 한다.
- 근거가 약한 결과는 `review_status="pending"`으로 내려가야 한다.
- CLI/ingestion 테스트는 Florence provider를 monkeypatch로 주입해도 기존 흐름이 깨지지 않아야 한다.

**Step 2: 실패 확인**

Run: `./.venv/bin/python -m pytest tests/test_florence2_tagger.py tests/test_ingestion_service.py tests/test_cli.py -q`

Expected:
- `ModuleNotFoundError` 또는 provider 분기 없음
- Florence-2 출력 파싱 관련 assertion 실패

### Task 2: Florence-2 로컬 태거 구현

**Files:**
- Create: `src/doc_finder/services/florence2_tagger.py`
- Modify: `src/doc_finder/services/tagging_service.py`
- Modify: `src/doc_finder/bootstrap.py`

**Step 1: Florence-2 전용 서비스 추가**

`src/doc_finder/services/florence2_tagger.py`에 `Florence2VisionTagger`를 구현한다.

구현 항목:
- 생성자 인자: `model_id`, `device`, `torch_dtype`, `max_new_tokens`, `num_beams`
- import 시점이 무거워지지 않도록 lazy model load
- 입력 이미지는 `PIL.Image.open(...).convert("RGB")`로 처리
- 이미지마다 `<OD>` 한 번, `<DETAILED_CAPTION>` 한 번 실행
- `<OD>` 결과는 `processor.post_process_generation(...)`로 파싱
- 아래 데이터를 합쳐 태그 후보 생성:
  - object detection 라벨
  - caption에서 추출한 명사/구 단위 후보
- 최종 반환값은 하나의 `TaggingResult`

**Step 2: 기존 서비스 계약 유지**

`src/doc_finder/services/tagging_service.py`에서는:
- `TaggingResult`, `TaggingError`, `StaticVisionTagger`, `HttpVisionTagger`는 유지
- 모든 provider가 같은 형태를 반환할 수 있도록 태그 문자열 정리용 작은 helper를 추가
- ingestion 로직은 이 파일로 옮기지 않고, Florence-2 모델 코드는 별도 파일에 둔다

**Step 3: provider 연결**

`src/doc_finder/bootstrap.py`에서:
- `DOC_FINDER_TAGGER_PROVIDER=florence2` 분기 추가
- 환경변수 추가:
  - `DOC_FINDER_FLORENCE2_MODEL_ID`
  - `DOC_FINDER_FLORENCE2_DEVICE`
  - `DOC_FINDER_FLORENCE2_TORCH_DTYPE`
  - optional `DOC_FINDER_FLORENCE2_MAX_NEW_TOKENS`
  - optional `DOC_FINDER_FLORENCE2_NUM_BEAMS`
- 기본 모델은 `microsoft/Florence-2-base`
- 기본 device 선택 규칙:
  - env가 있으면 그 값을 우선
  - 없으면 `cuda`, `mps`, `cpu` 순으로 선택
- 기본 dtype 선택 규칙:
  - `cuda`면 `float16`
  - 그 외는 `float32`

### Task 3: 후처리 규칙을 결정론적으로 정의

**Files:**
- Modify: `src/doc_finder/services/florence2_tagger.py`
- Modify: `src/doc_finder/services/query_normalizer.py`
- Test: `tests/test_florence2_tagger.py`

**Step 1: Florence-2 태그 추출 규칙 구현**

다음 규칙을 코드로 명시한다.
- 모든 태그는 소문자/trim 처리
- caption 결과에 쉼표 구분 표현이 있으면 candidate tag로 분리
- 2글자 미만 태그는 버리되, 이미 허용한 한글 명사는 예외 처리 가능하게 둔다
- 구두점만 있는 조각은 제거
- 순서를 유지한 채 중복 제거
- `keyword_tags`는 예를 들어 최대 12개까지만 유지

**Step 2: confidence 정책은 보수적으로**

Florence-2를 detector confidence처럼 다루면 안 된다. 1차는 heuristic로 계산한다.

규칙:
- 시작값 `0.45`
- `<OD>` 결과에 object label이 1개 이상 있으면 `+0.25`
- caption에서 추출한 태그와 OD 라벨이 겹치면 `+0.15`
- 태그 중 최소 1개가 `QueryNormalizer`를 통해 안정적으로 정규화되면 `+0.10`
- 최대값 `0.95`

review status:
- confidence `>= 0.80` 이면 `approved`
- 그 외는 `pending`

### Task 4: 런타임 의존성과 문서 정리

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `http/doc-finder.http`
- Create: `.env.example`

**Step 1: 런타임 의존성 추가**

`pyproject.toml`에 다음 의존성을 추가한다.
- `transformers`
- `torch`
- `pillow`

이 단계에서는 설치가 막히지 않는 한 optional acceleration 패키지는 추가하지 않는다.

**Step 2: Florence-2 실행 방법 문서화**

`README.md`에 아래를 추가한다.
- `static` / `http` / `florence2` 차이
- 로컬 `.env` 예시
- 첫 실행 시 모델 다운로드가 발생한다는 점
- CPU와 GPU 사용 시 체감 차이
- 예시 명령:
  - `source .env`
  - `python -m doc_finder.cli ingest ...`
  - `python -m doc_finder.cli search --query 사과 --top-k 5`

`.env.example` 예시:
- `DOC_FINDER_DATABASE_URL=postgresql://postgres:postgres@localhost:5431/doc_finder`
- `DOC_FINDER_TAGGER_PROVIDER=florence2`
- `DOC_FINDER_FLORENCE2_MODEL_ID=microsoft/Florence-2-base`
- `DOC_FINDER_FLORENCE2_DEVICE=cpu`
- `DOC_FINDER_FLORENCE2_TORCH_DTYPE=float32`

### Task 5: Florence-2 전체 경로 검증

**Files:**
- Test: `tests/test_florence2_tagger.py`
- Test: `tests/test_ingestion_service.py`
- Test: `tests/test_cli.py`

**Step 1: 자동 검증**

Run:
- `./.venv/bin/python -m pytest tests/test_florence2_tagger.py tests/test_ingestion_service.py tests/test_cli.py -q`
- `./.venv/bin/python -m pytest -q`

Expected:
- 모든 테스트 통과

**Step 2: 수동 스모크 검증**

`.env`를 로드한 상태에서 샘플 이미지 폴더로:
- `python -m doc_finder.cli ingest --image-dir /path/to/assets`
- `indexed_count >= 1` 확인
- `python -m doc_finder.cli search --query 사과 --top-k 5`
- 기대한 객체에 대해 `matched_tags` 또는 searchable normalized tag가 반환되는지 확인

**Step 3: 성능 sanity check**

로컬 장비에서 아래를 기록한다.
- 첫 모델 로드 시간
- 샘플 이미지 10장 기준 평균 태깅 시간

CPU가 너무 느리면 provider 구조는 유지하고, 설계를 바꾸는 대신 GPU 권장 사항만 문서화한다.

### Assumptions

- Florence-2는 로컬 전용 태거 provider이며 검색/인덱스 스키마는 이번 단계에서 바꾸지 않는다.
- 현재 `TaggingResult` 계약은 Florence-2 결과를 담기에 충분하다고 본다.
- 기본 모델은 `microsoft/Florence-2-base`로 고정한다.
- caption 파싱은 LLM 후처리가 아니라 결정론적 heuristic으로 처리한다.
- 이번 단계에서는 `.env` 자동 로드까지는 하지 않고, `.env.example`와 문서화까지만 포함한다.
