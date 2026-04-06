# DocFinder 프로젝트 요약본

## 1. 이 프로젝트가 뭐냐
**대량 일러스트 이미지에 구조화된 태그를 자동 생성·저장하고, LangGraph 기반 리트리버로 객체명 검색과 필터링을 가능하게 하는 시스템**이다.

이 프로젝트는 단일 기능이 아니다. 두 개의 축이 같이 있어야 성립한다.

- Ingestion Graph: 이미지 적재, 검증, 태그 생성, 정규화, 저장
- Retrieval Graph: 사용자 질의를 받아 검색 후보를 찾고 랭킹하는 검색 워크플로우

즉, `태그 저장 시스템`만도 아니고, `검색 엔진`만도 아니다.  
**태그 인덱스를 만들고, 그 위에서 실제 검색 UX까지 제공하는 시스템**이다.

이 시스템은 CMS를 대체하지 않는다.

- CMS/JSON: 원본 truth
- DocFinder: 태그 인덱스 + 검색 projection + 리트리버

---

## 2. 왜 필요한가
현재 문제는 아래 네 가지다.

- 이미지 수가 10만 개를 넘으면 수작업 태깅 비용이 너무 크다.
- 사람마다 태그 기준이 달라 일관성이 깨진다.
- 이미지와 콘텐츠가 연결돼 있어도 객체명으로 바로 찾기 어렵다.
- CMS에 원본 데이터는 있어도 검색용 인덱스와 검색 UX가 부족하다.

예를 들어 일러스트 안에 사과가 있으면 사용자가 `사과`라고 검색했을 때 바로 나와야 한다.  
지금 문서 기준으로 단순 태그 저장만 하면 이 경험이 약하다.  
그래서 **태그 생성/저장과 리트리버를 같이 설계해야 한다.**

---

## 3. 대표 사용자 시나리오

### 시나리오 A. 객체명으로 이미지 찾기
- 사용자가 `사과`를 검색한다.
- Retrieval Graph가 질의를 정규화한다.
- `normalized_tags` exact/synonym 후보를 찾는다.
- 저장된 tag projection 기반 임베딩 검색으로 추가 후보를 찾는다.
- 후보를 병합하고 랭킹한다.
- 결과로 `content_id + illustration_id + preview + tags + cms_ref`를 반환한다.

### 시나리오 B. 표기가 달라도 같은 이미지를 찾기
- 사용자가 `apple`을 검색한다.
- 시스템은 `사과`와 동일 normalized 개념으로 처리한다.
- 같은 이미지가 검색된다.

### 시나리오 C. 운영자가 태깅 품질을 관리하기
- 운영자는 low-confidence 이미지 목록을 본다.
- confidence가 낮은 태그만 review queue에서 확인한다.
- 모델 버전이 바뀌면 재태깅 job을 다시 실행한다.

---

## 4. 시스템이 어떻게 동작하냐

### 적재와 태깅
1. 이미지 자산을 배치로 수집한다.
2. 파일 크기, 포맷, 중복 여부를 검증한다.
3. 모델이 객체와 속성을 태그로 생성한다.
4. taxonomy 태그와 자유태그를 분리한다.
5. 검색용 normalized tag를 만든다.
6. confidence, 모델 버전, 참조 키와 함께 저장한다.

### 검색과 조회
1. 사용자가 `사과`, `고양이`, `버스` 같은 질의를 입력한다.
2. Retrieval Graph가 질의를 정규화한다.
3. exact/synonym tag 후보를 찾는다.
4. tag projection 임베딩 검색으로 추가 후보를 찾는다.
5. 후보를 병합하고 랭킹한다.
6. `content_id + illustration_id + preview + tags + cms_ref + score`를 반환한다.
7. 상세 콘텐츠는 CMS가 이어서 응답한다.

즉, 1차 MVP의 검색은 단순 태그 필터가 아니라 **LangGraph 기반 리트리버**가 중심이다.

---

## 5. 왜 FastAPI와 LangGraph냐

### FastAPI
- 배치 적재 job 생성
- 태그 조회 API
- 리트리버 검색 API
- 운영/모니터링용 엔드포인트 제공

### LangGraph
- Ingestion Graph와 Retrieval Graph를 각각 명시적으로 설계할 수 있다.
- 적재, 태그 생성, 정규화, 저장, review 분기를 구조적으로 표현할 수 있다.
- 질의 정규화, 후보 탐색, 후보 병합, 랭킹, 응답 조립을 구조적으로 표현할 수 있다.
- 디버깅과 재처리에 유리하다.

정리하면:

> **FastAPI는 외부 인터페이스, LangGraph는 내부 적재/검색 오케스트레이션 레이어다.**

---

## 6. 저장 모델 핵심
1차 저장 단위는 `content_id + illustration_id`다.

예시 필드:

- `content_id`
- `illustration_id`
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

중요한 점:

- CMS/JSON은 원본 데이터 저장소
- DocFinder는 검색용 projection 저장소

즉, 콘텐츠 본문이나 문제 원문을 DocFinder가 중복 저장하지 않는다.

---

## 7. 1차 MVP에서 꼭 들어가야 할 것

- 10만+ 일러스트 이미지 적재 가능한 배치 구조
- 500KB 이하 이미지 검증 규칙
- taxonomy + 자유태그 생성
- normalized tag 생성
- tag projection 생성
- `content_id + illustration_id` 기준 저장
- LangGraph Retrieval Graph 기반 객체명 검색 API
- CMS 참조 키 반환
- low-confidence review 분리

---

## 8. 1차 응답 계약
검색 결과는 상세 원문을 직접 다 들고 오지 않는다.

기본 응답:

- `content_id`
- `illustration_id`
- `thumbnail_url` 또는 preview path
- 대표 태그
- confidence
- retrieval score
- `cms_ref`

이 방식이 좋은 이유:

- CMS 중복 저장을 피한다.
- 동기화 리스크가 적다.
- 검색 시스템 책임이 명확하다.

---

## 9. 지금 기준 한 줄 결론
> **DocFinder는 대량 일러스트 이미지에 객체/속성 태그를 생성·저장하고, LangGraph 리트리버로 객체명 검색과 필터링을 가능하게 하는 태깅 + 검색 시스템이다.**

---

## 10. 후속 확장
이번 단계에서 중심이 아닌 항목은 후속으로 둔다.

- OCR 기반 검색
- PDF chunking
- 이미지 업로드 기반 유사 검색
- 고도화된 reranker
- 생성형 답변

즉, 지금은 **태그 생성·저장·리트리버 구축**이 핵심이고, 멀티모달 검색 엔진 전체는 그 다음 단계다.
