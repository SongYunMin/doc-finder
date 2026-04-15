# Tagger Provider Decoupling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Florence-2 전용 부트스트랩 결합을 제거하고, PaliGemma 2 같은 새 비전 모델 provider를 같은 프로젝트 안에서 독립적으로 추가할 수 있게 만든다.

**Architecture:** `ingestion`은 기존 `TaggingResult` 계약만 유지하고, provider 선택과 환경변수 해석은 공통 registry/factory 계층으로 이동한다. `bootstrap`은 더 이상 특정 모델 구현을 import 하지 않고, 각 provider 모듈이 자신만의 설정과 생성 로직을 가진다.

**Tech Stack:** Python 3.13, FastAPI, pytest, transformers

---

### Task 1: 공통 registry 테스트 작성

**Files:**
- Create: `tests/test_tagger_registry.py`
- Modify: `src/doc_finder/bootstrap.py`

**Step 1: Write the failing test**

```python
def test_build_tagger_from_env_supports_runtime_provider_registration(monkeypatch):
    ...
```

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_tagger_registry.py -q`
Expected: FAIL because `doc_finder.taggers` registry module does not exist yet.

**Step 3: Write minimal implementation**

공통 registry/factory를 추가하고, `DOC_FINDER_TAGGER_PROVIDER` 값을 기반으로 builder를 찾도록 구현한다.

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_tagger_registry.py -q`
Expected: PASS

### Task 2: bootstrap 의존성 분리

**Files:**
- Modify: `src/doc_finder/bootstrap.py`
- Modify: `tests/test_florence2_tagger.py`

**Step 1: Write the failing test**

```python
def test_build_default_ingestion_service_uses_generic_tagger_builder(monkeypatch):
    ...
```

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_tagger_registry.py::test_build_default_ingestion_service_uses_generic_tagger_builder -q`
Expected: FAIL because bootstrap still builds Florence-2 directly.

**Step 3: Write minimal implementation**

`bootstrap`이 공통 builder만 호출하도록 바꾸고, Florence 관련 env 해석은 provider 모듈로 이동한다.

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_tagger_registry.py::test_build_default_ingestion_service_uses_generic_tagger_builder -q`
Expected: PASS

### Task 3: provider 모듈 분리와 회귀 방지

**Files:**
- Create: `src/doc_finder/taggers/__init__.py`
- Create: `src/doc_finder/taggers/registry.py`
- Create: `src/doc_finder/taggers/providers/__init__.py`
- Create: `src/doc_finder/taggers/providers/http.py`
- Create: `src/doc_finder/taggers/providers/static.py`
- Create: `src/doc_finder/taggers/providers/florence2.py`
- Modify: `tests/test_ingestion_service.py`
- Test: `tests/test_florence2_tagger.py`

**Step 1: Write the failing test**

기존 Florence bootstrap 테스트를 provider registry 기준으로 바꿔 작성한다.

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_florence2_tagger.py tests/test_ingestion_service.py -q`
Expected: FAIL because provider wiring changed and tests are outdated.

**Step 3: Write minimal implementation**

provider별 생성 모듈을 분리하고, ingestion 테스트는 Florence 직접 의존 대신 일반 태거 계약 중심으로 정리한다.

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_tagger_registry.py tests/test_florence2_tagger.py tests/test_ingestion_service.py tests/test_cli.py -q`
Expected: PASS
