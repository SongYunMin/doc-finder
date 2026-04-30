# 로컬 doc-finder Gemma4 호출 명령어

작성일: 2026-04-30

로컬 Mac에서 `doc-finder`를 실행하고, 다른 서버 머신의 Gemma4 adapter를 호출하는 명령어만 정리한다.

## 1. 서버 연결 확인

`<gemma-machine-ip>`는 Gemma4 adapter가 떠 있는 서버 IP로 바꾼다.

```bash
curl http://<gemma-machine-ip>:18080/health
nc -vz <gemma-machine-ip> 18080
```

## 2. 불필요한 임시 이미지 제거

`images/5001_1069_payload.png`는 파일명 규칙에 맞지 않아 ingest에서 reject된다.

```bash
mkdir -p tmp
mv images/5001_1069_payload.png tmp/
```

## 3. SVG를 PNG로 변환

먼저 대상만 확인한다.

```bash
.venv/bin/python scripts/convert_svg_assets_to_png.py images
```

PNG 생성만 먼저 한다. SVG는 유지된다.

```bash
.venv/bin/python scripts/convert_svg_assets_to_png.py images --apply
```

PNG 결과를 확인한 뒤 SVG를 삭제한다.

```bash
.venv/bin/python scripts/convert_svg_assets_to_png.py images --apply --delete-source
```

## 4. PNG 흰 배경 적용

투명 PNG 대상만 확인한다.

```bash
.venv/bin/python scripts/apply_white_background_to_png_assets.py images
```

투명 PNG를 흰 배경 RGB PNG로 덮어쓴다.

```bash
.venv/bin/python scripts/apply_white_background_to_png_assets.py images --apply
```

## 5. 작은 샘플로 먼저 태깅

```bash
mkdir -p tmp/gemma-test
cp images/5001_1069.png tmp/gemma-test/5001_1069.png

env -u DOC_FINDER_DATABASE_URL \
  DOC_FINDER_TAGGER_PROVIDER=http \
  DOC_FINDER_VISION_ENDPOINT=http://<gemma-machine-ip>:18080/vision/tag \
  .venv/bin/python -m doc_finder.cli ingest --image-dir tmp/gemma-test
```

## 6. 전체 images 태깅

샘플 결과가 괜찮을 때만 전체를 돌린다.

```bash
env -u DOC_FINDER_DATABASE_URL \
  DOC_FINDER_TAGGER_PROVIDER=http \
  DOC_FINDER_VISION_ENDPOINT=http://<gemma-machine-ip>:18080/vision/tag \
  .venv/bin/python -m doc_finder.cli ingest --image-dir images
```

## 7. 전처리 확인 명령

단일 PNG가 흰 배경 payload로 바뀌는지 확인한다.

```bash
.venv/bin/python - <<'PY'
from io import BytesIO
from pathlib import Path
from PIL import Image
from doc_finder.services.tagging_service import _read_vision_payload_bytes

payload = _read_vision_payload_bytes(Path("images/5001_1069.png"))
img = Image.open(BytesIO(payload))
print("mode:", img.mode)
print("size:", img.size)
print("pixel_0_0:", img.getpixel((0, 0)))
PY
```

기대값:

```text
mode: RGB
pixel_0_0: (255, 255, 255)
```

## 8. 테스트

```bash
.venv/bin/python -m pytest tests/test_tagging_service.py tests/test_svg_asset_converter.py tests/test_white_background_asset_converter.py -q
.venv/bin/python -m pytest -q
```

## 9. 리스크

- `DOC_FINDER_VISION_ENDPOINT`에 `127.0.0.1`을 넣으면 로컬 Mac을 가리킨다. 서버 머신 IP를 넣어야 한다.
- 처음부터 전체 이미지를 돌리지 말고 1장 또는 20~50장 샘플로 먼저 확인한다.
- `tagging_failed`가 전부 뜨면 품질 문제가 아니라 endpoint 연결 실패일 가능성이 높다.
