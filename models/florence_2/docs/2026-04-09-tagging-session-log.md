# 2026-04-09 태깅 세션 로그

## 목적

이 문서는 이번 세션에서 진행한 이미지 태깅 관련 작업의 시행착오, 시도 방법, 질문, 결정사항을 정리한 기록이다.

중점 주제:
- 이미지 검색 MVP 정리
- `static` 태거 기반 스모크 테스트
- `Florence-2` 로컬 태거 통합
- SVG 지원 추가
- 현재 태깅 품질 한계 확인

---

## 이번 세션에서 확정한 전제

- CMS는 source of truth로 유지한다.
- 검색 시스템은 문항 전체 JSON이 아니라 참조 키와 태그 인덱스를 관리한다.
- 1차 검색은 문항 본문 검색이 아니라 이미지 태그 검색이다.
- 파일명 규칙은 기본적으로 `unitId_dataId_imageOrder`를 따르되, `imageOrder`가 없으면 `1`로 본다.
- 대량 이미지 원본은 프로젝트 내부에 넣지 않고 외부 경로를 참조하는 것이 맞다.
- 현재 데이터셋에는 일반 사물 일러스트와 문제 도식/기하 이미지가 섞여 있을 가능성이 높다.

---

## 주요 질문과 답

### 1. 검색 API가 문항 JSON까지 반환해야 하나?

결론:
- 1차는 아니라고 판단했다.
- 검색 API는 `unit_id`, `data_id`, `image_id`, `cms_ref` 중심으로 가고, 문항 JSON 조인은 후속 단계로 분리한다.

이유:
- 검색 책임과 CMS 책임을 분리하기 위해서다.
- 인덱스와 원본 데이터의 중복 소유를 피하기 위해서다.

### 2. 기준 식별자는 무엇인가?

결론:
- `unit_id + data_id + image_id`

배경:
- `unitId`는 문항 개념
- `dataId`는 같은 문항 개념 아래의 컨텐츠
- `imageOrder`는 같은 `unitId + dataId` 조합 내 이미지 순서

추가 변경:
- `10709_13048.svg`처럼 `imageOrder`가 없는 파일도 허용하고, 이 경우 `image_id=1`로 처리하도록 바꿨다.

### 3. 30만 장 이미지를 바로 넣으면 되나?

결론:
- 로직상은 경로만 넘기면 가능하지만, 프로젝트 폴더 내부에 넣는 건 비추천이다.
- 현재 MVP 구현을 그대로 30만 장에 바로 투입하는 것도 위험하다.

이유:
- repo가 너무 무거워진다.
- 단일 프로세스/단건 순회/단건 저장 구조라 운영성 보강이 더 필요하다.

---

## 구현/설정 시도 기록

### A. 초기 `static` 태거 기반 E2E 확인

시도:
- `static-tags.json`과 샘플 이미지로 `ingest -> search`를 검증했다.

문제:
- `static-tags.json`를 이미지 폴더 아래에 두면 같이 스캔되어 `invalid_filename` reject가 발생했다.
- JSON 키와 실제 파일명 확장자(`.svg` vs `.png`)가 다르면 `No static tags configured ...` 오류가 발생했다.
- `DOC_FINDER_DATABASE_URL`이 빠지면 `InMemory` 저장소로 떨어져 `ingest`와 `search`가 서로 다른 프로세스에서 빈 메모리를 보게 됐다.

조치:
- `static-tags.json`는 별도 폴더로 분리
- JSON 키를 실제 파일명과 정확히 맞춤
- DB URL을 같은 셸에서 명시적으로 로드

결과:
- `static` 모드에서는 샘플 PNG 기준 `ingest`와 `search`가 정상 동작했다.

### B. Postgres 포트 문제

상황:
- `5431`로 포트를 바꿨는데 연결이 계속 실패했다.

원인:
- Docker 매핑이 `5431 -> 5431`로 잘못 잡혀 있었고, 컨테이너 내부 Postgres는 `5432`에서만 리슨하고 있었다.

조치:
- 컨테이너를 `5431 -> 5432`로 다시 띄웠다.

결과:
- `postgresql://postgres:postgres@localhost:5431/doc_finder`로 정상 접속됐다.

### C. 검색 결과가 비어 있거나 이상하게 나오는 문제

상황:
- `search` 결과가 `[]`였다가, 또 어떤 경우에는 태그가 없는데도 결과가 나왔다.

원인 1:
- 실제 `image_index` row가 0건이었다.

원인 2:
- 현재 검색 그래프는 exact tag hit가 없어도 semantic fallback 후보를 그대로 결과에 남긴다.
- 그래서 `matched_tags=[]`인데도 약한 semantic score로 결과가 노출됐다.

해석:
- 이건 “태그가 맞아서 나온 결과”가 아니라 semantic fallback 후보다.

판단:
- 지금 단계에서는 검색보다 태그 품질이 우선이다.
- 나중에 `matched_tags=[]` semantic-only 결과를 숨기거나 threshold를 강화할 필요가 있다.

---

## Florence-2 통합 기록

### 1. 왜 Florence-2를 붙였나?

배경:
- 외부 비전 API 비용 없이 로컬 태깅을 시도하기 위해서다.
- `static/http` provider 구조를 유지한 채 `florence2` provider를 추가했다.

추가한 것:
- `Florence2VisionTagger`
- `DOC_FINDER_TAGGER_PROVIDER=florence2`
- Florence-2 관련 환경변수
- 태그 후처리 규칙

### 2. Florence-2 초기 호환성 문제

문제:
- `transformers` 최신 버전에서 Florence-2 remote code와 호환성 이슈가 반복 발생했다.

대표 증상:
- `forced_bos_token_id`
- `_supports_sdpa`
- `dtype` / `torch_dtype`
- `past_key_values` 관련 generate 오류

조치:
- `transformers`를 `4.49.0`으로 맞췄다.
- `attn_implementation="eager"`를 넣었다.
- 모델 로더 인자를 `torch_dtype` 기준으로 정리했다.

결과:
- Florence-2 모델 로드와 추론 경로가 동작하기 시작했다.

### 3. Florence-2 추가 의존성 문제

문제:
- `einops`, `timm`가 없어서 모델 코드 import 단계에서 실패했다.

조치:
- 두 패키지를 설치하고 의존성 목록에 반영했다.

결과:
- 모델 로딩 실패는 해결됐다.

### 4. SVG 입력 문제

문제:
- 메타데이터 검증 단계는 SVG를 허용했지만, Florence 입력 단계에서 `PIL.Image.open()`이 SVG를 직접 열 수 없었다.

조치:
- `cairosvg`를 이용해 `SVG -> PNG bytes -> PIL RGB 이미지` 변환 경로를 추가했다.
- 테스트도 추가했다.

추가 문제:
- `cairosvg`만 설치해서는 안 되고 시스템 `cairo` 라이브러리도 필요했다.

macOS 조치:
- `brew install cairo`

결과:
- SVG도 Florence-2 태깅 경로로 들어갈 수 있게 됐다.

### 5. Florence-2 실제 태그 품질 문제

관찰:
- 일반 사물 이미지가 아닌 문제 도식/기하 이미지에서 Florence-2가 매우 부정확했다.

실제 오탐 예시:
- 이미지: 기하 도형 문제 그림
- Florence 결과: `person`, `wheelchair`, `grass`, `black`, `white`

판단:
- 이건 단순한 오차가 아니라 데이터 도메인과 태거 방식이 안 맞는 상태다.
- Florence-2를 전체 데이터셋 주력 태거로 바로 가는 건 재검토가 필요하다.

추가 관찰:
- Florence-2는 긴 문장형 캡션을 내놓는 경우가 많았다.
- 후처리로 문장을 잘게 줄여 `person`, `wheelchair`, `grass` 같은 토큰으로 축약하는 로직을 추가했다.
- 그래도 기하/도식 이미지에는 여전히 잘 맞지 않는다.

---

## WD tagger vs Florence-2 논의

질문:
- `WD tagger`가 더 나을 수 있나?

판단:
- 현재 샘플이 수학/기하 도식 이미지라면, `WD`도 근본 해결책이 아닐 가능성이 높다.

이유:
- `WD tagger`는 Danbooru/애니 일러스트 태그 편향이 강하다.
- 지금 필요한 태그는 `사각형`, `대각선`, `각도`, `36도`, `55도`, `점 A/B/C/D` 같은 문제 구조 태그다.
- 일반 객체 태거나 보루 태거 둘 다 방향이 다르다.

결론:
- 데이터셋이 일반 사물 일러스트와 문제 도식 이미지로 나뉜다면, 태깅 전략도 분기해야 한다.

---

## 현재 코드에서 바뀐 핵심 포인트

- `florence2` provider 추가
- Florence 태그 후처리 추가
- `ingest` 콘솔 로그 추가
  - `[indexed] ... keyword_tags=... normalized_tags=...`
  - `[duplicate] ...`
  - `[reject] ...`
- SVG rasterize 지원 추가
- 파일명에서 `imageOrder`가 없으면 `1`로 처리

---

## 실제로 해본 디버깅/확인 명령들

아래는 이번 세션에서 반복적으로 사용한 확인 명령들이다.

### DB 상태 확인

```bash
./.venv/bin/python - <<'PY'
import os
import psycopg

with psycopg.connect(os.environ["DOC_FINDER_DATABASE_URL"], autocommit=True) as conn:
    with conn.cursor() as cur:
        cur.execute("select count(*) from image_index")
        print("image_index =", cur.fetchone()[0])
        cur.execute("select count(*) from image_rejects")
        print("image_rejects =", cur.fetchone()[0])
PY
```

### 최근 reject 확인

```bash
./.venv/bin/python - <<'PY'
import os
import psycopg

with psycopg.connect(os.environ["DOC_FINDER_DATABASE_URL"], autocommit=True) as conn:
    with conn.cursor() as cur:
        cur.execute("""
            select asset_path, reason, detail
            from image_rejects
            order by recorded_at desc
            limit 20
        """)
        for row in cur.fetchall():
            print(row)
PY
```

### 태거 직접 호출

```bash
./.venv/bin/python - <<'PY'
from pathlib import Path
from doc_finder.bootstrap import _build_default_tagger
from doc_finder.services.query_normalizer import QueryNormalizer

tagger = _build_default_tagger(query_normalizer=QueryNormalizer())
result = tagger.tag(Path("images/10565_20077_1.png"), "debug-sha")
print(result)
PY
```

### 전체 인덱스 초기화

```bash
./.venv/bin/python - <<'PY'
import os
import psycopg

with psycopg.connect(os.environ["DOC_FINDER_DATABASE_URL"], autocommit=True) as conn:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE image_index, image_rejects RESTART IDENTITY")
        print("truncated")
PY
```

---

## 이번 세션의 결론

### 해결된 것

- `static` 태거 기준 E2E 동작 확인
- Postgres 포트/환경변수 문제 해결
- Florence-2 provider 구조 통합
- Florence-2 실행 호환성 문제 일부 해결
- SVG를 Florence-2 태깅 경로에 올리는 것까지 확인
- `imageOrder`가 없는 파일명 허용
- `ingest` 콘솔 로그 추가

### 아직 해결되지 않은 것

- 문제 도식/기하 이미지에 대한 태깅 품질
- Florence-2 결과가 서비스 태그로 바로 쓰일 만큼 안정적인지
- semantic fallback 검색이 너무 느슨한 문제

### 현재 가장 중요한 판단

- 지금 데이터셋에 문제 도식 이미지가 많이 섞여 있다면, Florence-2를 전체 주력 태거로 가는 방향은 재검토가 필요하다.
- 일반 사물 일러스트와 문제 도식 이미지를 분기해서 태깅 전략을 가져가는 것이 더 현실적이다.

---

## 다음 세션에서 논의할 것

- 데이터셋을 일반 사물 일러스트 / 문제 도식 이미지로 어떻게 구분할지
- 도식 이미지 전용 태깅 전략을 OCR + 규칙 기반으로 설계할지
- semantic fallback 검색을 당분간 끌지 여부
- Florence-2를 유지한다면 어떤 이미지 유형에만 적용할지
