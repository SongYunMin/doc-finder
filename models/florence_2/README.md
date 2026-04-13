# models/florence_2

Florence-2 전용 문서와 실험 기록을 모아두는 작업 루트다.

## 목적

- Florence-2 관련 실험과 자산을 다른 모델 작업과 분리한다.
- 공용 입력 이미지는 루트 `images/`에서 공유한다.
- Florence-2 실구현은 `src/doc_finder/models/florence_2/` 아래에 둔다.
- 이 디렉터리는 Python import 대상이 아니라 작업 메모와 산출물 정리용이다.

## 현재 구성

- `../../static-tags.json`: 정적 태거 실험용 공용 태그 fixture
- `docs/2026-04-09-florence2-local-tagger.md`: Florence-2 로컬 태거 구현 계획
- `docs/2026-04-09-tagging-session-log.md`: Florence-2 태깅 시행착오와 품질 이슈 기록

## 공용 입력 이미지

여러 모델이 같은 이미지를 재사용할 수 있도록 샘플 이미지는 루트 `images/` 디렉터리에 둔다.
