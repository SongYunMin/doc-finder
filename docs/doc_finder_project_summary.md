# DocFinder 프로젝트 요약본

## 1. 이 프로젝트가 뭐냐
**대량 일러스트 이미지에 구조화된 태그를 자동 생성·저장하고, 객체명 검색과 필터링이 가능하도록 인덱스를 구축하는 시스템**이다.

핵심은 이미지 자체를 그냥 보관하는 것이 아니라, 다음 정보를 함께 만드는 데 있다.

- taxonomy 기반 분류 태그
- 자유 키워드 태그
- 검색용 normalized tag
- confidence
- CMS 참조 키

이 시스템은 CMS를 대체하지 않는다.

- CMS/JSON: 원본 truth
- DocFinder: 태그 인덱스 + 검색 projection 저장소

---

## 2. 왜 필요한가
현재 문제는 단순하다.

- 이미지가 많아질수록 수작업 태깅 비용이 커진다.
- 태그 품질과 일관성이 사람마다 달라진다.
- 이미지와 콘텐츠는 연결돼 있어도, 객체명으로 바로 찾기 어렵다.
- CMS에는 원본 데이터가 있어도 검색용 메타데이터 인덱스가 부족하다.

예를 들어 일러스트 안에 사과가 있어도 사용자가 `사과`라고 검색했을 때 바로 해당 이미지를 찾지 못하면 운영 효율이 크게 떨어진다.

즉, 이 프로젝트는 단순 저장 시스템이 아니라 **이미지 메타데이터를 자동으로 생성하고 검색 가능한 형태로 만드는 인덱싱 시스템**이다.

---

## 3. 대표 사용자 시나리오

### 시나리오 A. 객체명으로 이미지 찾기
- 사용자가 `사과`를 검색한다.
- 시스템은 `normalized_tags`에서 `사과`, `apple` 같은 정규화된 태그를 기준으로 검색한다.
- 결과로 `content_id + illustration_id + 미리보기 + 대표 태그`를 반환한다.
- 상세 콘텐츠 정보는 CMS에서 다시 조회한다.

### 시나리오 B. 비슷한 객체를 일관되게 찾기
- 사용자가 `고양이`를 검색한다.
- 시스템은 taxonomy 태그와 자유태그를 함께 활용해 관련 일러스트를 찾는다.
- low-confidence 태그는 검수 대상으로 따로 관리한다.

### 시나리오 C. 운영자가 태깅 품질을 관리하기
- 운영자는 모델이 생성한 태그와 confidence를 본다.
- confidence가 낮은 이미지만 review queue에서 다시 검토한다.
- 모델 버전이 바뀌면 재태깅 job을 돌릴 수 있다.

---

## 4. 시스템이 어떻게 동작하냐

### 적재와 태깅
1. 이미지 자산을 배치로 수집한다.
2. 이미지 크기, 포맷, 파일 크기를 검증한다.
3. 모델이 이미지에서 객체와 속성을 태그로 생성한다.
4. taxonomy 태그와 자유태그를 분리한다.
5. 검색용 normalized tag를 만든다.
6. confidence, 모델 버전, 참조 키와 함께 저장한다.

### 검색과 조회
1. 사용자가 `사과` 같은 객체명을 입력한다.
2. 질의를 normalized tag 기준으로 정규화한다.
3. 태그 인덱스에서 일치하는 이미지를 조회한다.
4. `content_id + illustration_id + preview + tags + cms_ref`를 반환한다.
5. 상세 데이터가 필요하면 CMS가 이어서 응답한다.

즉, 1차 MVP의 검색은 벡터 검색보다 **태그 인덱스 검색**이 중심이다.

---

## 5. 왜 FastAPI와 LangGraph냐

### FastAPI
- 배치 적재 job 생성
- 태그 조회 API
- 태그 기반 검색 API
- 운영/모니터링용 엔드포인트 제공

### LangGraph
- 태깅 파이프라인을 단계별로 명시적으로 제어
- 검증, 태그 생성, 정규화, 저장, review 분기를 구조적으로 표현
- 디버깅과 재처리에 유리

정리하면:

> **FastAPI는 외부 인터페이스, LangGraph는 내부 태깅 오케스트레이션 레이어다.**

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
- confidence 저장
- `content_id + illustration_id` 기준 저장
- 객체명 검색 API
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
- `cms_ref`

이 방식이 좋은 이유:

- CMS 중복 저장을 피한다.
- 동기화 리스크가 적다.
- 검색 시스템 책임이 명확하다.

---

## 9. 지금 기준 한 줄 결론
> **DocFinder는 대량 일러스트 이미지에 객체/속성 태그를 생성·저장하고, 객체명 검색과 필터링을 가능하게 하는 LangGraph 기반 태깅 인덱스 시스템이다.**

---

## 10. 후속 확장
이번 단계에서 중심이 아닌 항목은 후속으로 둔다.

- OCR 기반 검색
- PDF chunking
- 이미지-텍스트 하이브리드 retrieval
- 벡터 검색과 reranker
- 생성형 답변

즉, 지금은 **태그 생성·저장·조회**가 핵심이고, 멀티모달 검색 엔진 전체는 그 다음 단계다.
