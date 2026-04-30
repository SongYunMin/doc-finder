# Gemma4 서버 머신 CLI

작성일: 2026-04-30

Gemma4/Ollama가 설치된 서버 머신에서 실행하는 명령어다.

## 1. Ollama 확인

```bash
ollama list
curl http://127.0.0.1:11434/api/tags
hostname -I
```

`gemma4:31b`가 보여야 한다. `hostname -I` 출력 중 로컬에서 접근 가능한 IP를 로컬 문서의 `<gemma-machine-ip>`에 넣는다.

## 2. 태깅 서버 실행

```bash
export DOC_FINDER_OLLAMA_URL=http://127.0.0.1:11434
export DOC_FINDER_OLLAMA_MODEL=gemma4:31b
export DOC_FINDER_OLLAMA_TIMEOUT_SECONDS=120

python -m doc_finder.tagger_server.main --host 0.0.0.0 --port 18080
```

## 3. 서버 머신에서 확인

```bash
curl http://127.0.0.1:18080/health
```

## 4. 단일 이미지로 Ollama 직접 확인

이 명령은 태깅 서버를 거치지 않는다. 모델 smoke test 용도다.

```bash
IMG=$(base64 < 5001_1069.png | tr -d '\n')

curl -X POST http://127.0.0.1:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4:31b",
    "messages": [{
      "role": "user",
      "content": "이미지를 검색용 태그로 분석해라. 반드시 JSON object 하나만 반환한다. 필드: keyword_tags, normalized_tags, confidence, review_status.",
      "images": ["'"$IMG"'"]
    }],
    "format": "json",
    "stream": false,
    "think": false,
    "options": {"temperature": 0}
  }'
```

## 5. 리스크

- Ollama `11434` 포트를 공개망에 직접 열지 않는다.
- 외부에는 태깅 서버 `18080`만 열고, 가능하면 사설망/방화벽 allowlist/SSH 터널을 사용한다.
- 여러 이미지를 한 요청에 넣지 않는다. 이미지 1장당 요청 1번이 품질과 디버깅에 유리하다.
