# doc-finder

Gemma4/Ollama 기반 이미지 태깅 API 서버와 이미지 전처리 도구다.

현재 `main`은 검색/DB/색인 코드를 제거하고, Gemma 태깅 실험에 필요한 최소 구성만 남긴다. 이전 구조는 `archive/doc-finder-current` 브랜치에 백업되어 있다.

## Structure

- `src/doc_finder/tagger_server/`: Gemma4 태깅 FastAPI 서버
- `src/doc_finder/services/image_payload.py`: SVG/투명 이미지 흰 배경 PNG 변환 helper
- `src/doc_finder/services/svg_asset_converter.py`: SVG 파일 일괄 PNG 변환
- `src/doc_finder/services/white_background_asset_converter.py`: 투명 PNG 일괄 흰 배경 적용
- `scripts/`: 전처리 실행 스크립트
- `docs/`: 서버 머신/로컬 실행 명령어

## Install

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Run Gemma Tagger Server

Gemma/Ollama가 설치된 서버 머신에서 실행한다.

```bash
export DOC_FINDER_OLLAMA_URL=http://127.0.0.1:11434
export DOC_FINDER_OLLAMA_MODEL=gemma4:31b
export DOC_FINDER_OLLAMA_TIMEOUT_SECONDS=120

python -m doc_finder.tagger_server.main --host 0.0.0.0 --port 18080
```

Endpoints:

- `GET /health`
- `POST /vision/tag`

Request:

```json
{
  "filename": "5001_1069.png",
  "sha256": "sha-value",
  "content_base64": "..."
}
```

Response:

```json
{
  "keyword_tags": ["geometry", "rectangular prism"],
  "normalized_tags": ["직육면체", "입체도형"],
  "confidence": 0.95,
  "review_status": "approved"
}
```

## Preprocess Assets

SVG를 흰 배경 PNG로 변환한다.

```bash
python scripts/convert_svg_assets_to_png.py images
python scripts/convert_svg_assets_to_png.py images --apply
```

투명 PNG를 흰 배경 RGB PNG로 덮어쓴다.

```bash
python scripts/apply_white_background_to_png_assets.py images
python scripts/apply_white_background_to_png_assets.py images --apply
```

## Test

```bash
.venv/bin/python -m pytest -q
```
