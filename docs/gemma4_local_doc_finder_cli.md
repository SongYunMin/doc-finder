# 로컬 Gemma4 태깅 호출 명령어

작성일: 2026-04-30

로컬 Mac에서 이미지를 전처리하고, 서버 머신의 Gemma4 태깅 서버를 호출하는 명령어다.

## 1. 서버 연결 확인

```bash
curl http://<gemma-machine-ip>:18080/health
nc -vz <gemma-machine-ip> 18080
```

## 2. SVG를 PNG로 변환

```bash
.venv/bin/python scripts/convert_svg_assets_to_png.py images
.venv/bin/python scripts/convert_svg_assets_to_png.py images --apply
```

PNG 확인 후 SVG를 삭제하려면:

```bash
.venv/bin/python scripts/convert_svg_assets_to_png.py images --apply --delete-source
```

## 3. PNG 흰 배경 적용

```bash
.venv/bin/python scripts/apply_white_background_to_png_assets.py images
.venv/bin/python scripts/apply_white_background_to_png_assets.py images --apply
```

## 4. 단일 이미지 태깅 호출

```bash
IMG=$(base64 < images/5001_1069.png | tr -d '\n')

curl -X POST http://<gemma-machine-ip>:18080/vision/tag \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "5001_1069.png",
    "sha256": "manual-test",
    "content_base64": "'"$IMG"'"
  }'
```

## 5. 테스트

```bash
.venv/bin/python -m pytest tests/test_tagger_server.py tests/test_svg_asset_converter.py tests/test_white_background_asset_converter.py -q
.venv/bin/python -m pytest -q
```

## 6. 리스크

- `<gemma-machine-ip>`에 `127.0.0.1`을 넣으면 로컬 Mac을 가리킨다. 서버 머신 IP를 넣어야 한다.
- 처음부터 전체 이미지를 돌리지 말고 1장 또는 20~50장 샘플로 먼저 확인한다.
- 태깅 실패가 전부 뜨면 품질 문제가 아니라 endpoint 연결 실패일 가능성이 높다.
