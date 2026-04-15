# florence_2_tuning

## 목적

이 폴더는 `models/finetune_florence_2/`에 정리한 준비 문서를 실제 실행 가능한 학습 파이프라인으로 연결하는 초안이다.

- 런타임 태거 구현은 계속 `src/doc_finder/models/florence_2/`에 둔다.
- 이 폴더는 Florence-2 GeoTag 튜닝 실행 코드와 결과 산출물 경로를 맡는다.
- 현재 기본 전략은 `full fine-tuning + vision encoder freeze`다.

## 현재 구성

- `dataset.py`: JSONL 학습 데이터 로더
- `metrics.py`: exact match / tag precision / recall / F1 계산
- `training.py`: Florence-2 모델 로드, collator, 학습 루프, 검증, 체크포인트 저장

## 데이터 포맷

기본 입력 포맷은 기존 준비 폴더의 템플릿을 그대로 따른다.

- 샘플 파일: `models/finetune_florence_2/templates/geotag_dataset.sample.jsonl`
- 필수 필드:
  - `image`
  - `prompt`
  - `target_text`
  - `split`

예시:

```json
{"image":"images/10565_20077_1.png","prompt":"<GeoTag>","target_text":"triangle; point_a; point_b; point_c; angle_label","split":"train"}
```

## 실행 예시

스모크 런은 작은 subset으로 먼저 검증하는 것이 안전하다.

```bash
./.venv/bin/python -m models.florence_2_tuning.training \
  --dataset /Users/knowre-yunmin/doc-finder/models/finetune_florence_2/templates/geotag_dataset.sample.jsonl \
  --image-root /Users/knowre-yunmin/doc-finder \
  --output-dir /Users/knowre-yunmin/doc-finder/models/florence_2_tuning/runs/smoke \
  --epochs 1 \
  --batch-size 1 \
  --validation-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --train-limit 1 \
  --validation-limit 1
```

## 주의사항

- 현재 repo에는 gold dataset이 아직 고정되지 않았으므로, 이 코드는 학습 파이프라인 검증용 성격이 강하다.
- `models/finetune_florence_2/README.md`에 적어둔 `1~5단계`가 끝나기 전에는 본학습을 돌리면 안 된다.
- GPU 메모리가 부족하면 다음 단계에서 PEFT/LoRA를 추가하는 것이 맞다. 지금 바로 넣지 않은 이유는 현재 의존성과 검증 범위를 불필요하게 키우지 않기 위해서다.
