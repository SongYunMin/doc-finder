# Gemma4 서버 머신 CLI 테스트

작성일: 2026-04-30

Gemma4/Ollama가 설치된 서버 머신에서 직접 실행하는 명령어만 정리한다.

## 1. Ollama 상태 확인

```bash
ollama list
curl http://127.0.0.1:11434/api/tags
hostname -I
```

`gemma4:31b`가 보여야 한다. `hostname -I` 출력 중 로컬 PC에서 접근 가능한 IP를 로컬 문서의 `<gemma-machine-ip>`에 넣는다.

## 2. 단일 이미지 직접 테스트

서버 머신에 테스트 이미지가 있다고 가정한다.

```bash
IMG=$(base64 < 5001_1069.png | tr -d '\n')

curl -X POST http://127.0.0.1:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4:31b",
    "messages": [{
      "role": "user",
      "content": "이미지를 검색용 태그로 분석해라. 반드시 JSON object 하나만 반환한다. 필드: keyword_tags, normalized_tags, confidence, review_status. review_status는 approved 또는 pending 중 하나만 사용한다.",
      "images": ["'"$IMG"'"]
    }],
    "format": "json",
    "stream": false,
    "think": false,
    "options": {
      "temperature": 0
    }
  }'
```

주의: 이 직접 호출은 `doc-finder` 전처리를 타지 않는다. 투명 PNG/SVG 품질 확인은 로컬 전처리 후 보내는 방식이 더 정확하다.

## 3. Adapter 실행

서버 머신에서 `doc-finder` 요청을 받아 Ollama로 전달하는 임시 adapter다. 로컬 PC가 이 서버의 `18080` 포트로 접근할 수 있어야 한다.

```bash
OLLAMA_URL=http://127.0.0.1:11434 OLLAMA_MODEL=gemma4:31b python - <<'PY'
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import urllib.request

OLLAMA_URL = os.environ["OLLAMA_URL"].rstrip("/")
MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:31b")

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers["content-length"])
        payload = json.loads(self.rfile.read(length))

        prompt = (
            "이미지를 검색용 태그로 분석해라. "
            "반드시 JSON object 하나만 반환한다. "
            "Markdown 코드블록, 설명, 중복 JSON, 기타 텍스트를 출력하지 않는다. "
            "필드: keyword_tags, normalized_tags, confidence, review_status. "
            "review_status는 approved 또는 pending 중 하나만 사용한다."
        )
        ollama_payload = {
            "model": MODEL,
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [payload["content_base64"]]
            }],
            "format": "json",
            "stream": False,
            "think": False,
            "options": {"temperature": 0}
        }

        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=json.dumps(ollama_payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as res:
            ollama_response = json.loads(res.read())

        result = json.loads(ollama_response["message"]["content"])
        body = json.dumps(result, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

HTTPServer(("0.0.0.0", 18080), Handler).serve_forever()
PY
```

## 4. 서버 머신에서 포트 확인

```bash
curl http://127.0.0.1:18080/health
```

외부 PC에서 접근이 안 되면 서버 방화벽 또는 네트워크 설정을 확인한다.

## 5. 리스크

- Ollama `11434` 포트는 공개망에 직접 열지 않는다.
- 외부에는 adapter 포트만 열고, 가능하면 사설망/방화벽 allowlist/SSH 터널을 사용한다.
- 여러 이미지를 한 요청에 넣지 않는다. 이미지 1장당 요청 1번이 품질과 디버깅에 유리하다.
