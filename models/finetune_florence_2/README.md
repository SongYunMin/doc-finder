# finetune_florence_2

Hugging Face의 Florence-2 fine-tuning 예제를 현재 프로젝트의 `기하 이미지 -> 표준 태그 문자열` 작업으로 바꿔 적용하기 위한 준비 폴더다.

## 목표

- Florence-2 DocVQA 예제를 그대로 따라 하지 않고, 현재 검색 시스템에 맞는 `GeoTag` 학습 흐름으로 재해석한다.
- 실제 학습 전에 필요한 라벨 체계, 데이터셋 포맷, 평가 순서를 문서와 템플릿으로 고정한다.
- 런타임 앱 코드는 `src/doc_finder/models/florence_2/`에 두고, 이 폴더는 실험 문서와 준비물만 보관한다.

## 현재 구성

- `docs/2026-04-13-roadmap.md`: 단계별 전체 로드맵
- `docs/2026-04-13-hf-blog-adaptation.md`: HF 블로그 예제를 현재 목표에 맞게 바꾼 메모
- `templates/geotag_dataset.sample.jsonl`: 학습 데이터 샘플
- `templates/label_taxonomy.example.yaml`: 라벨 taxonomy 예시
- `templates/phase_checklist.md`: 단계별 체크리스트

## 권장 단계 수

현재 목표 기준 권장 단계는 총 `8단계`다.

1. 작업 정의와 출력 계약 고정
2. 태그 taxonomy 설계
3. 골드 라벨 데이터셋 구축
4. 데이터 분할과 품질 검수
5. zero-shot 베이스라인 측정
6. 학습 포맷 변환과 실행 환경 준비
7. 소규모 smoke fine-tuning
8. 본 fine-tuning 및 오프라인 평가

## 지금 바로 해야 하는 단계

아직 해야 할 것은 `1~5단계`다.  
이 단계가 끝나기 전에는 본 학습에 들어가면 안 된다.

이유:
- 라벨 체계가 없으면 모델이 무엇을 맞혀야 하는지 정의되지 않는다.
- 평가셋이 없으면 튜닝 성공 여부를 판단할 수 없다.
- 현재 Florence-2가 기하 이미지에서 얼마나 틀리는지 baseline을 모르면 학습 효과를 비교할 수 없다.

## 참고 자료

- Hugging Face blog: `Fine-tuning Florence-2`
- Hugging Face Transformers docs: `Florence-2`
- Hugging Face PEFT docs: `LoRA`
