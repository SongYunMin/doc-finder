# 프로젝트 초안 문서
## 대량 일러스트 이미지 태그 생성 및 LangGraph 리트리버 시스템 (Working Title: DocFinder)

- 문서 버전: v0.3 Draft
- 문서 성격: 프로젝트 개요 + 제품 요구사항 + 기술 설계 초안
- 기준: 현재 대화에서 합의된 방향을 반영한 2차 재작성본

---

## 1. 문서 목적

이 문서는 다음 목적을 위해 작성한다.

1. 프로젝트의 1차 목표를 `이미지 태그 생성 및 저장 + 사용자 검색` 중심으로 명확히 정의한다.
2. CMS 원본 데이터와 DocFinder 인덱스의 역할을 분리한다.
3. 일러스트 이미지 태깅 파이프라인과 Retrieval Graph를 구현 가능한 수준으로 정리한다.
4. LangGraph를 Ingestion Graph와 Retrieval Graph의 오케스트레이션 레이어로 사용하는 이유를 명확히 한다.
5. 이후 구현, 운영, 평가 문서의 기준점으로 사용한다.

이 문서는 단순 아이디어 메모가 아니다. 바로 구현과 기술 설계 검토에 사용할 수 있어야 한다.

---

## 2. 프로젝트 한 줄 정의

**10만 개 이상의 일러스트 이미지에 구조화된 태그를 자동 생성·저장하고, LangGraph 기반 리트리버로 객체명 검색과 필터링이 가능하도록 만드는 시스템을 구축한다.**

핵심은 단순 저장이 아니다. 다음 다섯 가지를 한 번에 해결하는 것이다.

- 대량 이미지 배치 적재
- taxonomy + 자유태그 생성
- 검색용 normalized tag 저장
- CMS와 연결되는 검색 projection 구축
- 사용자 질의를 처리하는 Retrieval Graph 구축

---

## 3. 문제 정의

현재 해결하려는 문제는 다음과 같다.

### 3.1 사용자가 원하는 것

사용자 또는 운영자는 아래와 같은 요구를 가진다.

- `사과`, `고양이`, `버스` 같은 객체명으로 일러스트를 자연스럽게 찾고 싶다.
- 같은 콘텐츠 안에 이미지가 여러 장 있어도 정확한 이미지 단위로 찾고 싶다.
- 태그 결과를 CMS와 연결해 후속 조회나 노출에 활용하고 싶다.
- 사람이 수작업으로 태그를 붙이지 않아도 대량 이미지를 일관되게 분류하고 싶다.

### 3.2 기존 접근의 문제

- CMS/JSON에 원본 데이터는 있어도 검색용 태그 인덱스가 없다.
- 수작업 태깅은 비용이 크고, 사람마다 태그 기준이 달라 일관성이 깨진다.
- 이미지와 콘텐츠가 연결돼 있어도 객체명으로 검색되지 않는 경우가 많다.
- 동일한 개념도 `사과`, `apple`, `빨간 사과`처럼 여러 표기로 흩어진다.
- 단순 exact tag 검색만으로는 사용자 검색 UX가 약하다.
- 이미지 수가 10만 개를 넘으면 수동 정제와 운영 검수가 병목이 된다.

### 3.3 핵심 결론

이 프로젝트는 단순 CMS 보조 기능이 아니다. **일러스트 이미지용 태그 인덱스를 만들고, 그 위에서 LangGraph 기반 검색 UX를 제공하는 시스템**이다.

---

## 4. 프로젝트 목표

### 4.1 최상위 목표

일러스트 이미지에 객체/속성 태그를 자동 생성하고 저장하고, 사용자가 객체명으로 이미지를 찾을 수 있는 LangGraph 리트리버를 만든다.

### 4.2 세부 목표

1. 10만 개 이상의 일러스트 이미지를 배치로 적재할 수 있어야 한다.
2. 각 이미지에 대해 taxonomy 태그와 자유태그를 생성할 수 있어야 한다.
3. `사과`와 `apple` 같은 표기 차이를 흡수할 normalized tag를 저장해야 한다.
4. 검색용 tag text projection과 임베딩 후보를 만들 수 있어야 한다.
5. 검색 결과는 `unit_id + data_id + image_id + preview + tags + cms_ref + score`를 반환해야 한다.
6. low-confidence 태그는 운영 검수 대상으로 분리할 수 있어야 한다.
7. CMS/JSON은 원본 truth로 유지하고, DocFinder는 searchable projection만 관리해야 한다.
8. 태깅 파이프라인과 리트리버 파이프라인은 LangGraph로 명시적으로 제어 가능해야 한다.

---

## 5. 비목표(지금 하지 않을 것)

초기 단계에서 아래 항목은 범위에서 제외한다.

1. OCR 중심 검색
2. PDF chunking
3. 이미지-텍스트 하이브리드 retrieval
4. 생성형 답변
5. 범용 에이전트 시스템
6. CMS 전체 데이터를 대체 저장하는 구조

핵심 원칙은 명확하다.

> **1차 목적은 태그 생성·저장과 사용자 검색 UX를 함께 성립시키는 것이다.**

---

## 6. 대상 데이터

현재 가정하는 1차 데이터는 다음과 같다.

- 일러스트 기반 이미지
- 10만 개 이상
- 파일당 500KB 이하
- 기존 JSON/CMS와 연결 가능한 이미지 자산

### 6.1 데이터의 구조적 특성

- 하나의 콘텐츠에 여러 장의 일러스트가 연결될 수 있다.
- 이미지 자체는 작지만 개수가 매우 많다.
- 사용자 검색은 본문보다 객체명과 속성 중심으로 일어난다.
- 원본 콘텐츠 정보는 CMS가 authoritative source다.
- 인덱싱 시스템은 이미지 검색에 필요한 최소 메타데이터와 참조 키만 저장해야 한다.

### 6.2 기본 식별 전략

1차 표준 식별자는 아래 조합으로 고정한다.

- `unit_id`
- `data_id`
- `image_id`

파일명 규칙은 `<unitId>_<dataId>_<imageOrder>.svg|png` 로 본다. `unitId` 는 문항 개념, `dataId` 는 같은 문항 개념 아래의 컨텐츠, `imageOrder` 는 같은 `unitId + dataId` 안에서의 이미지 순서다.

`문제 번호만`으로 식별하는 방식은 비추천이다. 같은 문항 개념에 다른 컨텐츠와 여러 이미지가 붙을 수 있어 충돌과 추적 문제가 생기기 쉽다.

---

## 7. 핵심 요구사항

### 7.1 기능 요구사항

#### FR-1. 이미지 적재
- 시스템은 대량 이미지를 배치 단위로 적재할 수 있어야 한다.
- 적재 시 파일 크기, 포맷, 중복 여부를 검증해야 한다.

#### FR-2. 태그 생성
- 각 이미지에 대해 객체/속성 태그를 생성할 수 있어야 한다.
- 태그는 taxonomy 태그와 자유태그를 함께 지원해야 한다.

#### FR-3. normalized tag 생성
- 저장된 태그는 검색용 normalized tag로 정규화되어야 한다.
- 예: `사과`, `apple`은 동일한 normalized 개념으로 매핑될 수 있어야 한다.

#### FR-4. 태그 저장
- 태그와 함께 confidence, 모델 버전, 생성 시각, 검수 상태를 저장해야 한다.

#### FR-5. 객체명 검색
- 사용자는 `사과`, `고양이`, `버스` 같은 객체명으로 이미지를 검색할 수 있어야 한다.
- 1차 검색은 `정확 태그 + 동의어/표기 정규화 + 임베딩 기반 후보 보강`까지 포함한다.

#### FR-6. LangGraph 리트리버
- 검색은 단순 태그 필터가 아니라 LangGraph 기반 Retrieval Graph로 구성되어야 한다.
- Retrieval Graph는 질의 정규화, 태그 후보 탐색, 임베딩 후보 탐색, 후보 병합, 랭킹, 응답 조립 단계를 포함해야 한다.

#### FR-7. CMS 참조 반환
- 검색 결과는 CMS를 다시 조회할 수 있는 참조 키를 반환해야 한다.
- 검색 시스템은 콘텐츠 상세를 직접 소유하지 않는다.

#### FR-8. 재태깅
- 모델 버전 변경이나 정책 변경 시 이미지 재태깅이 가능해야 한다.

#### FR-9. review queue
- confidence가 낮은 이미지는 검수 대상으로 분리할 수 있어야 한다.

### 7.2 비기능 요구사항

#### NFR-1. 대량 처리
- 10만 개 이상 데이터를 배치로 처리할 수 있어야 한다.

#### NFR-2. 일관성
- 동일 이미지 또는 동일 개념에 대해 태그 결과가 일관되어야 한다.

#### NFR-3. 검색 사용성
- 사용자가 exact tag를 몰라도 객체명 검색이 실용적으로 동작해야 한다.

#### NFR-4. 재처리 가능성
- 적재와 태깅은 재실행 가능해야 하며, 중복 적재를 방지해야 한다.

#### NFR-5. 관측 가능성
- job 상태, 성공/실패 수, confidence 분포, 모델 버전별 결과를 추적할 수 있어야 한다.
- 검색 단계에서는 질의 정규화, 후보 수, 랭킹 결과, 최종 점수를 추적할 수 있어야 한다.

#### NFR-6. 시스템 책임 분리
- CMS는 원본 truth, DocFinder는 searchable projection 저장소라는 책임이 분명해야 한다.

---

## 8. 대표 사용자 시나리오

### 시나리오 A. 사과가 포함된 일러스트 찾기
사용자 입력: `사과`

기대 결과:

- 사과가 포함된 일러스트가 반환된다.
- 검색 결과에는 `unit_id`, `data_id`, `image_id`, 대표 태그, preview 정보가 포함된다.
- 상세 화면은 CMS 참조 키를 이용해 다시 조회한다.
- 단순 exact match가 아니라 관련도가 높은 결과가 상위에 노출된다.

### 시나리오 B. 영어 표기로도 같은 결과 찾기
사용자 입력: `apple`

기대 결과:

- `사과`와 동일한 normalized tag로 처리되어 동일 이미지가 검색된다.

### 시나리오 C. 느슨한 표현으로도 찾기
사용자 입력: `빨간 과일`

기대 결과:

- tag exact match가 아니어도 임베딩 후보 탐색을 통해 사과 이미지가 상위 후보에 포함된다.

### 시나리오 D. 운영자가 저신뢰 결과 검수하기
운영자 입력: `review_status = pending`

기대 결과:

- low-confidence 태그가 붙은 이미지만 검수 큐에서 확인할 수 있다.

### 시나리오 E. 콘텐츠 단위가 아닌 이미지 단위 식별
사용자 입력: 콘텐츠 내부 이미지 중 특정 객체 검색

기대 결과:

- `image_id` 기준으로 정확한 이미지가 구분된다.
- 같은 `unit_id + data_id`에 여러 이미지가 있어도 혼동하지 않는다.

---

## 9. 핵심 설계 원칙

### 원칙 1. CMS는 원본 truth다
DocFinder는 CMS를 대체하지 않는다.

- CMS/JSON: 콘텐츠 원문, 문제 정보, 상세 메타데이터
- DocFinder: 태그, 검색 projection, 검색 응답에 필요한 최소 정보

### 원칙 2. 검색 인덱스는 projection 저장소다
DocFinder는 이미지 검색에 필요한 정보만 저장한다.

- 이미지 식별자
- 참조 키
- 태그
- confidence
- 모델 버전
- preview 정보
- 검색용 projection 텍스트
- 임베딩 벡터

### 원칙 3. 태그는 3층 구조로 관리한다
1차 태그 구조는 아래와 같다.

- `taxonomy_tags`
- `keyword_tags`
- `normalized_tags`

이 구조가 있어야 운영 일관성과 검색 편의성을 동시에 확보할 수 있다.

### 원칙 4. 검색은 normalized tag만으로 끝나지 않는다
1차 MVP의 검색 출발점은 `normalized_tags`지만, 검색 UX는 그것만으로 해결하지 않는다.

- `사과`
- `apple`
- 복수형, 표기 변형
- 느슨한 서술형 질의

이 차이를 흡수하기 위해 Retrieval Graph는 태그 후보와 임베딩 후보를 함께 다뤄야 한다.

### 원칙 5. LangGraph는 Ingestion과 Retrieval을 모두 제어한다
LangGraph는 범용 agent가 아니라 두 개의 핵심 워크플로우를 통제하기 위한 레이어다.

- Ingestion Graph: 검증, 태그 생성, 정규화, 저장, review 분기
- Retrieval Graph: 질의 정규화, 후보 탐색, 후보 병합, 랭킹, 응답 조립

---

## 10. 권장 기술 스택

현재 대화 기준 권장안은 아래와 같다.

### 10.1 백엔드
- FastAPI
- Python

### 10.2 오케스트레이션
- LangGraph

### 10.3 태그 생성
- 이미지 태그 생성 모델 또는 vision-language model
- taxonomy 매핑용 규칙 또는 후처리 레이어

### 10.4 저장소
- 메타데이터 저장소: relational DB 우선
- 원본 자산: CMS 또는 기존 asset storage
- 검색 projection 저장: metadata DB 중심

### 10.5 검색 보강
- tag text projection 기반 임베딩
- lightweight rerank 또는 score fusion

### 10.6 참조 시스템
- CMS/JSON

---

## 11. LangGraph를 유지하는 이유

이 프로젝트의 핵심은 단순 모델 호출이 아니라 **대량 태깅과 사용자 검색을 모두 안정적으로 제어하는 것**이다.

### LangGraph를 유지하는 이유
- 이미지 검증, 태그 생성, 정규화, 저장을 단계별로 명시할 수 있다.
- 실패 지점 추적이 쉽다.
- low-confidence 분기와 재시도 전략을 구조적으로 넣기 좋다.
- job 단위 디버깅과 운영 로그에 유리하다.
- 검색 단계에서도 질의 정규화, 후보 탐색, 후보 병합, 랭킹을 명시적으로 제어할 수 있다.

정리하면,

> **이 프로젝트는 태깅 인덱스와 검색 UX를 함께 설계해야 하며, 두 흐름 모두 LangGraph로 제어하는 것이 맞다.**

---

## 12. 전체 아키텍처

```text
[Client / Operator]
  ├─ ingest request
  ├─ object-name search query
  └─ image tag inspection

[API Layer - FastAPI]
  ├─ ingest endpoint
  ├─ search endpoint
  ├─ tag lookup endpoint
  └─ job status endpoint

[Ingestion Graph - LangGraph]
  ├─ load_image
  ├─ validate_image
  ├─ generate_visual_tags
  ├─ map_taxonomy_tags
  ├─ generate_keyword_tags
  ├─ normalize_tags
  ├─ build_tag_projection
  ├─ score_confidence
  ├─ persist_tags
  └─ enqueue_review_if_needed

[Retrieval Graph - LangGraph]
  ├─ normalize_query
  ├─ search_normalized_tags
  ├─ search_embedding_candidates
  ├─ merge_candidates
  ├─ rerank_or_score_fuse
  ├─ attach_cms_ref
  └─ format_response

[Storage]
  ├─ CMS / JSON: source of truth
  ├─ Tag Store / Metadata DB: searchable projection
  └─ Asset path reference: image location
```

---

## 13. 데이터 적재 및 태깅 설계

### 13.1 단계 개요

1. 이미지 수집
2. 파일 검증
3. 중복 확인
4. 태그 생성
5. taxonomy 매핑
6. 자유태그 생성
7. normalized tag 생성
8. tag projection 생성
9. confidence 계산
10. 저장
11. review queue 분기

### 13.2 검증 규칙

- 파일 크기 500KB 이하
- 지원 포맷만 허용
- 해상도/비율이 극단적으로 깨진 경우 reject 또는 별도 처리
- 동일 해시 중복 적재 방지

### 13.3 태그 생성 원칙

- taxonomy 태그는 운영과 필터링 기준으로 사용한다.
- 자유태그는 표현력과 검색 recall 보강 용도로 사용한다.
- normalized tag는 검색 인덱스 기준값이다.
- tag projection은 임베딩 검색을 위한 요약 텍스트다.

### 13.4 normalized tag 예시

```json
{
  "keyword_tags": ["사과", "빨간 사과", "apple"],
  "normalized_tags": ["사과"],
  "tag_text_projection": "fruit apple red apple illustration"
}
```

---

## 14. 저장 모델 설계

### 14.1 저장 단위
저장 단위는 이미지 1개다.

기본 키:

- `unit_id`
- `data_id`
- `image_id`

### 14.2 저장 예시

```json
{
  "unit_id": 10565,
  "data_id": 20077,
  "image_id": 1,
  "object_path": "cms://assets/10565/20077/1.png",
  "sha256": "9b3d...",
  "file_size": 412321,
  "width": 512,
  "height": 512,
  "taxonomy_tags": ["fruit", "food"],
  "keyword_tags": ["사과", "apple", "빨간 과일"],
  "normalized_tags": ["사과"],
  "tag_text_projection": "fruit apple red apple illustration",
  "embedding_vector": [0.12, -0.33, 0.88],
  "tag_scores": {
    "사과": 0.94,
    "fruit": 0.89
  },
  "model_version": "vision-tagger-v1",
  "tagged_at": "2026-04-06T10:00:00Z",
  "review_status": "approved",
  "cms_ref": {
    "unit_id": 10565,
    "data_id": 20077
  }
}
```

### 14.3 필수 저장 필드

- `unit_id`
- `data_id`
- `image_id`
- `object_path`
- `sha256`
- `file_size`
- `width`
- `height`
- `taxonomy_tags`
- `keyword_tags`
- `normalized_tags`
- `tag_text_projection`
- `embedding_vector`
- `tag_scores`
- `model_version`
- `tagged_at`
- `review_status`

---

## 15. 검색 및 조회 설계

### 15.1 1차 검색 방식

1차 MVP의 검색은 아래 Retrieval Graph 흐름으로 고정한다.

```text
query
  → normalize query
  → search normalized_tags
  → search embedding candidates
  → merge candidates
  → optional filter apply
  → rerank or score fuse
  → response format
```

### 15.2 검색 범위

1차 MVP에서 지원하는 범위:

- 정확 태그 검색
- 동의어/표기 정규화 검색
- 태그 projection 기반 임베딩 후보 탐색
- 태그 후보 + 임베딩 후보 병합

후속 확장:

- 이미지 업로드 기반 검색
- OCR/텍스트 검색
- 더 강한 reranker

### 15.3 응답 계약

검색 결과는 아래 수준까지만 반환한다.

- `unit_id`
- `data_id`
- `image_id`
- `thumbnail_url` 또는 preview path
- 대표 태그
- confidence
- retrieval score
- `cms_ref`

상세 콘텐츠 정보는 CMS에서 다시 조회한다.

### 15.4 검색 예시

#### 입력

```json
{
  "query": "사과",
  "filters": {
    "review_status": "approved"
  },
  "top_k": 5
}
```

#### 응답

```json
{
  "results": [
    {
      "unit_id": 10565,
      "data_id": 20077,
      "image_id": 1,
      "thumbnail_url": "cms://thumbs/10565/20077/1.png",
      "tags": ["사과", "fruit"],
      "confidence": 0.94,
      "score": 0.91,
      "cms_ref": {
        "unit_id": 10565,
        "data_id": 20077
      }
    }
  ]
}
```

---

## 16. API 초안

### 16.1 적재 API

#### `POST /ingest/images`
- 배치 태깅 job 생성

요청 예시:

```json
{
  "source": "cms-export-2026-04-06.json",
  "batch_size": 1000
}
```

### 16.2 job 조회 API

#### `GET /ingest/jobs/{job_id}`
- 진행률
- 성공/실패 수
- 에러 사유

### 16.3 태그 조회 API

#### `GET /images/{unit_id}/{data_id}/{image_id}/tags`
- 저장된 태그, confidence, 모델 버전 조회

### 16.4 검색 API

#### `POST /search/by-tags`
- `query`, `filters`, `top_k` 입력
- Retrieval Graph가 `normalized_tags + embedding candidates`를 함께 처리

---

## 17. 평가 설계

감으로 성공 여부를 판단하면 안 된다. 1차는 태그 생성과 검색 정확도를 동시에 평가해야 한다.

### 17.1 핵심 평가 케이스

- `사과` 검색 시 사과가 포함된 일러스트가 반환된다.
- `apple` 검색 시 동일 일러스트가 동의어 정규화로 반환된다.
- `사과 그림`, `빨간 과일` 같은 질의에도 관련 이미지가 상위 후보에 포함된다.
- 같은 `unit_id + data_id` 아래에 이미지가 여러 장 있어도 `image_id` 기준으로 정확히 구분된다.
- 500KB 초과 이미지는 reject 또는 별도 처리 정책이 적용된다.
- confidence가 낮은 태그는 review 대상으로 분리된다.
- 검색 결과만으로는 CMS를 대체하지 않고, 상세는 CMS에서 재조회한다.

### 17.2 평가 지표

- 태그 정확도
- taxonomy 일치율
- normalized tag 일치율
- 검색 Precision@k
- 검색 Recall@k
- low-confidence 비율
- job 처리량

---

## 18. 개발 우선순위 및 단계별 계획

### Phase 1. 태그 인덱스 MVP
목표: 대량 이미지에 태그를 생성하고 LangGraph 리트리버로 객체명 검색이 가능한 최소 기능 완성

포함 범위:

- 이미지 적재
- 검증
- 태그 생성
- taxonomy 매핑
- normalized tag 저장
- tag projection 생성
- Retrieval Graph 기반 검색 API
- CMS 참조 반환

### Phase 2. 운영 안정화
목표: 태그 품질과 재처리 안정성 강화

포함 범위:

- review queue
- confidence 정책 개선
- 재태깅
- 중복 처리
- 태그 정책 보정

### Phase 3. 검색 고도화
목표: 1차 리트리버에서 멀티모달 검색으로 확장

포함 범위:

- 이미지 업로드 기반 검색
- OCR/텍스트 확장
- 더 강한 reranker

---

## 19. 주요 리스크 및 대응 전략

### 리스크 1. 태그 품질 불안정
문제:

- 모델이 객체를 과하게 일반화하거나 놓칠 수 있다.

대응:

- taxonomy + 자유태그 분리 저장
- confidence 저장
- review queue 운영

### 리스크 2. 동의어/표기 불일치
문제:

- `사과`, `apple`, `빨간 사과`가 따로 놀 수 있다.

대응:

- normalized tag 계층 도입
- 질의 정규화 규칙 운영

### 리스크 3. 리트리버 사용성 저하
문제:

- exact tag만 보면 사용자가 원하는 결과를 못 찾을 수 있다.

대응:

- tag projection 임베딩 후보 탐색
- 후보 병합
- lightweight rerank 또는 score fusion

### 리스크 4. CMS와 인덱스 불일치
문제:

- CMS 원본과 태그 인덱스 간 연결이 어긋날 수 있다.

대응:

- `unit_id + data_id + image_id`를 표준 키로 고정
- projection 저장소 역할만 수행

### 리스크 5. 대량 배치 운영 부담
문제:

- 10만+ 이미지 처리 시 배치 실패와 재시도 관리가 중요하다.

대응:

- job 단위 상태 저장
- 실패 이미지 분리
- 재처리 가능 구조 확보

---

## 20. 구현 체크리스트

### 데이터/적재
- [ ] 이미지 자산 목록 확보
- [ ] `unit_id + data_id + image_id` 기준 정리
- [ ] 파일 검증 규칙 구현
- [ ] 중복 방지 규칙 구현

### 태깅
- [ ] taxonomy 태그 정책 정의
- [ ] 자유태그 생성 규칙 정의
- [ ] normalized tag 규칙 정의
- [ ] tag projection 규칙 정의
- [ ] confidence 저장 구조 마련

### 검색
- [ ] `/search/by-tags` API 구현
- [ ] normalized tag 검색 구현
- [ ] tag projection 임베딩 검색 구현
- [ ] 후보 병합 및 score 정렬 구현
- [ ] 필터 처리 구현
- [ ] CMS 참조 응답 형식 고정

### 운영
- [ ] review queue 기준 정의
- [ ] job 상태 추적 설계
- [ ] 재태깅 전략 정의

### 평가
- [ ] 객체명 검색 정답셋 작성
- [ ] 태그 품질 평가 스크립트 구현
- [ ] 검색 정확도 지표 정리

---

## 21. 현재 시점의 권장 MVP 결정안

현재 대화 기준 가장 현실적인 시작점은 아래와 같다.

- 백엔드: FastAPI
- 오케스트레이션: LangGraph
- 태그 생성: vision tagger 또는 VLM
- 태그 구조: taxonomy + 자유태그 + normalized tag
- 검색 보강: tag projection 임베딩 + lightweight rerank
- 참조 시스템: CMS/JSON
- 저장소 역할: searchable projection 중심

### 이유

- 이미지 수가 많아도 구조적으로 운영 가능하다.
- CMS와 역할 충돌이 적다.
- `사과` 같은 객체명 검색 요구를 직접 충족한다.
- 이후 이미지 업로드 기반 검색으로 확장할 수 있다.

---

## 22. 의사결정 로그 요약

현재 기준으로 합의된 내용은 아래와 같다.

1. 1차 목적은 `일러스트 태그 생성 및 저장 + 사용자 검색`이다.
2. 검색은 LangGraph 리트리버를 활용한 객체명 검색이 중심이다.
3. `사과` 검색은 반드시 지원해야 하는 대표 시나리오다.
4. 태그 스키마는 `taxonomy + 자유태그 + normalized_tags` 혼합형으로 간다.
5. CMS/JSON은 원본 truth로 유지하고, DocFinder는 projection 저장소로 동작한다.
6. 기본 식별 키는 `unit_id + data_id + image_id`다.
7. LangGraph는 Ingestion Graph와 Retrieval Graph를 함께 통제하는 레이어로 유지한다.

---

## 23. 최종 정리

이 프로젝트의 본질은 아래 한 문장으로 요약된다.

> **DocFinder는 대량 일러스트 이미지에 객체/속성 태그를 생성·저장하고, LangGraph 리트리버로 객체명 검색과 필터링을 가능하게 하는 태깅 + 검색 시스템이다.**

잘못된 방향은 분명하다.

- CMS를 대체 저장소로 만들려는 것
- 이미지 상세 원문을 DocFinder에 중복 저장하는 것
- 문제 번호만으로 식별하려는 것
- exact tag match만으로 검색 UX를 해결하려는 것
- OCR/PDF 검색을 1차 목적과 섞어 버리는 것

올바른 방향도 분명하다.

- 태그 생성과 저장을 1차 목표로 둘 것
- Retrieval Graph를 사용자 검색의 중심으로 둘 것
- normalized tag로 객체명 검색을 가능하게 할 것
- CMS는 truth, DocFinder는 projection으로 역할을 나눌 것
- `unit_id + data_id + image_id`를 표준 키로 유지할 것

이 원칙만 흔들리지 않으면 1차 구현 방향은 맞다.
