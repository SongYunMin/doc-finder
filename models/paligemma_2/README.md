# paligemma_2

PaliGemma 2 zero-shot 태거 실험 문서 폴더다.

## Runtime namespace

- 런타임 코드는 `src/doc_finder/models/paligemma_2/`에 둔다.
- provider 등록은 `src/doc_finder/taggers/providers/paligemma2.py`에서 한다.

## Default checkpoint

- `google/paligemma2-3b-mix-448`

## Preview example

```bash
python -m doc_finder.cli tag \
  --image-dir ./images \
  --tagger-provider paligemma2 \
  --paligemma2-model-id google/paligemma2-3b-mix-448
```
