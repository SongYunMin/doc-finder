# Florence-2 Layout Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Florence-2 전용 코드를 모델 네임스페이스와 루트 전용 폴더로 분리하고, 공용 입력 이미지는 루트 `images/` 디렉터리로 옮긴다.

**Architecture:** Python 코드의 실구현은 `src/doc_finder/models/florence_2` 네임스페이스를 기준으로 두고, `doc_finder.services.florence2_tagger`는 호환 shim으로 유지한다. 루트 `models/florence_2`는 문서와 실험 기록만 보관하고, 공용 입력 이미지는 루트 `images/`, 정적 태그 fixture는 루트 `static-tags.json`에 둔다.

**Tech Stack:** Python 3.13, pytest, setuptools package layout, argparse CLI

---

### Task 1: 새 Florence-2 모듈 경로를 테스트로 고정

**Files:**
- Modify: `tests/test_florence2_tagger.py`

**Step 1: Write the failing test**

`doc_finder.models.florence_2` 경로에서 `Florence2VisionTagger`를 import 할 수 있어야 하고, 기존 서비스 모듈은 같은 클래스를 재노출해야 한다.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_florence2_tagger.py -q`
Expected: 실구현이 루트 `models`를 바라봐서 FAIL

**Step 3: Write minimal implementation**

`src/doc_finder/models/florence_2/`에 구현을 두고, 루트 `models/florence_2/`는 문서 전용으로 정리한다.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_florence2_tagger.py -q`
Expected: PASS

### Task 2: Florence 전용 루트 폴더와 공용 이미지 루트를 정리

**Files:**
- Create: `models/florence_2/README.md`
- Move: `assets/tag/static-tags.json` -> `static-tags.json`
- Move: `assets/img/*` -> `images/*`
- Modify: `README.md`

**Step 1: Update references**

문서와 예시 경로를 새 구조에 맞게 바꾼다.

**Step 2: Keep compatibility**

런타임 동작은 경로 하드코딩이 없도록 유지하고, 예시와 자산만 새 기준으로 정리한다.

**Step 3: Verify layout**

Run: `find florence-2 -maxdepth 3 -type f | sort`
Expected: Florence 전용 자산과 문서가 보인다.

### Task 3: 전체 회귀 확인

**Files:**
- Modify: `src/doc_finder/bootstrap.py`
- Modify: `tests/test_ingestion_service.py`
- Modify: `.env.example`

**Step 1: Align imports and examples**

부트스트랩과 테스트가 새 모듈 경로를 기준으로 이해되도록 맞춘다.

**Step 2: Run focused tests**

Run: `pytest tests/test_florence2_tagger.py tests/test_ingestion_service.py tests/test_cli.py -q`
Expected: PASS

**Step 3: Run broader sanity check**

Run: `pytest -q`
Expected: 기존 사용자 변경과 무관한 범위에서 전체 PASS
