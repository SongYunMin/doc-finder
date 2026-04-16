# Florence-2 Fine-tuning Prep Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Hugging Face Florence-2 fine-tuning 블로그를 현재 프로젝트의 기하 태그 학습 목적에 맞게 해석한 준비 폴더와 단계 가이드를 만든다.

**Architecture:** 런타임 앱 코드는 건드리지 않고, 루트 `models/finetune_florence_2` 아래에 문서, 데이터셋 템플릿, 실험 체크리스트를 둔다. 문서는 `DocVQA` 예제를 그대로 복제하지 않고, 현재 목표인 `이미지 -> 표준 태그 문자열` 작업으로 재해석한다.

**Tech Stack:** Markdown, YAML, JSONL templates, Hugging Face Florence-2 references

---

### Task 1: 단계 체계와 산출물 정의

**Files:**
- Create: `models/finetune_florence_2/README.md`
- Create: `models/finetune_florence_2/docs/2026-04-13-roadmap.md`

**Step 1: 목표를 단계별로 정리한다**

최소 단계는 아래 흐름을 포함해야 한다.
- 문제 정의
- 태그 체계 정의
- 데이터 라벨링
- 데이터 분할
- 베이스라인 측정
- 소규모 학습
- 본 학습
- 앱 통합 판단

**Step 2: 각 단계의 종료 조건을 적는다**

각 단계마다 “다음 단계로 넘어가도 되는 기준”을 명시한다.

### Task 2: 블로그 적응 메모 작성

**Files:**
- Create: `models/finetune_florence_2/docs/2026-04-13-hf-blog-adaptation.md`

**Step 1: 원문 예제와 현재 목표의 차이를 명시한다**

다음 차이를 반드시 적는다.
- 블로그는 `DocVQA`
- 현재 목표는 `GeoTag`
- 질문-답변 학습이 아니라 표준 태그 문자열 생성

**Step 2: 그대로 가져갈 것과 버릴 것을 구분한다**

예:
- 가져갈 것: image + prompt -> text target 학습 구조
- 조정할 것: prompt prefix, 데이터 포맷, 평가 지표
- 경계할 것: DocVQA metric, 질문 기반 정답 포맷

### Task 3: 데이터 준비 템플릿 생성

**Files:**
- Create: `models/finetune_florence_2/templates/geotag_dataset.sample.jsonl`
- Create: `models/finetune_florence_2/templates/label_taxonomy.example.yaml`
- Create: `models/finetune_florence_2/templates/phase_checklist.md`

**Step 1: 데이터 샘플 포맷을 만든다**

필드는 최소한 아래를 포함한다.
- `image`
- `prompt`
- `target_text`
- `split`
- `notes`

**Step 2: 라벨 taxonomy 예시를 만든다**

기하 도형/관계/텍스트 라벨 카테고리를 나눠서 기록한다.

### Task 4: 검토와 전달

**Files:**
- Review only

**Step 1: 새 폴더만 검토한다**

Run: `find models/finetune_florence_2 -maxdepth 3 -type f | sort`
Expected: roadmap, adaptation memo, templates가 보인다.

**Step 2: 사용자에게 단계 수와 우선순위를 설명한다**

현재 목표에 필요한 총 단계 수와 “바로 해야 할 단계”를 구분해 설명한다.
