from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from pathlib import Path
import re

from doc_finder.services.query_normalizer import QueryNormalizer
from doc_finder.services.tagging_service import TaggingError, TaggingResult, clean_tag_candidates

_CAPTION_STOP_PREFIXES = (
    "the image",
    "this image",
    "the photo",
    "photo of",
    "the scene",
    "with the",
    "showing",
    "shows",
    "there is",
    "there are",
    "giving it",
)

_CAPTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "bottom",
    "classic",
    "feel",
    "giving",
    "image",
    "in",
    "is",
    "it",
    "night",
    "of",
    "photo",
    "shows",
    "the",
    "timeless",
    "visible",
    "with",
}


class Florence2VisionTagger:
    def __init__(
        self,
        model_id: str,
        device: str,
        torch_dtype: str,
        query_normalizer: QueryNormalizer,
        max_new_tokens: int = 512,
        num_beams: int = 3,
        prompt_runner: Callable[[str, Path], object] | None = None,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._torch_dtype = torch_dtype
        self._query_normalizer = query_normalizer
        self._max_new_tokens = max_new_tokens
        self._num_beams = num_beams
        self._prompt_runner = prompt_runner
        self._model = None
        self._processor = None

    def tag(self, asset_path: Path, sha256: str) -> TaggingResult:
        # 운영 경로와 진단 경로가 같은 프롬프트 조합을 보도록 OD와 caption을 함께 호출한다.
        try:
            od_result = self._run_prompt("<OD>", asset_path)
            caption_result = self._run_prompt("<CAPTION>", asset_path)
        except Exception as exc:  # noqa: BLE001
            raise TaggingError(f"Florence-2 tagging failed for {asset_path.name}.") from exc

        od_tags = self._extract_od_tags(od_result)
        caption_tags = self._extract_caption_tags(caption_result)
        keyword_tags = clean_tag_candidates([*od_tags, *caption_tags], max_tags=12)
        normalized_tags = self._build_normalized_tags(keyword_tags)
        confidence = self._compute_confidence(od_tags, caption_tags, normalized_tags)
        review_status = "approved" if confidence >= 0.80 else "pending"

        return TaggingResult(
            keyword_tags=keyword_tags,
            normalized_tags=normalized_tags,
            confidence=confidence,
            review_status=review_status,
        )

    def _run_prompt(self, task_prompt: str, asset_path: Path) -> object:
        if self._prompt_runner is not None:
            return self._prompt_runner(task_prompt, asset_path)

        model, processor = self._get_model_bundle()
        return _run_florence_prompt(
            model=model,
            processor=processor,
            task_prompt=task_prompt,
            asset_path=asset_path,
            device=self._device,
            torch_dtype=self._torch_dtype,
            max_new_tokens=self._max_new_tokens,
            num_beams=self._num_beams,
        )

    def _get_model_bundle(self) -> tuple[object, object]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        dtype = getattr(torch, self._torch_dtype)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True,
        ).to(self._device)
        self._processor = AutoProcessor.from_pretrained(
            self._model_id,
            trust_remote_code=True,
        )
        return self._model, self._processor

    def _extract_od_tags(self, payload: object) -> list[str]:
        if isinstance(payload, dict):
            od_payload = payload.get("<OD>", payload)
            labels = od_payload.get("labels", []) if isinstance(od_payload, dict) else []
            return clean_tag_candidates(labels)
        return []

    def _extract_caption_tags(self, payload: object) -> list[str]:
        if isinstance(payload, dict):
            caption_text = payload.get("<CAPTION>", "")
        elif isinstance(payload, str):
            caption_text = payload
        else:
            caption_text = ""

        raw_candidates = []
        for chunk in re.split(r"[,;\n\.]+", str(caption_text)):
            cleaned = chunk.strip()
            if cleaned:
                raw_candidates.extend(self._normalize_caption_chunk(cleaned))
        return clean_tag_candidates(raw_candidates)

    def _build_normalized_tags(self, keyword_tags: list[str]) -> list[str]:
        normalized_tags: list[str] = []
        for tag in keyword_tags:
            normalized_tags.extend(self._query_normalizer.normalize_tag_candidates(tag))
        return clean_tag_candidates(normalized_tags)

    def _compute_confidence(
        self,
        od_tags: list[str],
        caption_tags: list[str],
        normalized_tags: list[str],
    ) -> float:
        confidence = 0.45
        if od_tags:
            confidence += 0.25
        if set(od_tags).intersection(caption_tags):
            confidence += 0.15
        if normalized_tags and normalized_tags != clean_tag_candidates(caption_tags):
            confidence += 0.10
        return round(min(confidence, 0.95), 2)

    def _normalize_caption_chunk(self, chunk: str) -> list[str]:
        lowered = " ".join(chunk.lower().split())
        if not lowered:
            return []

        if self._looks_like_object_phrase(lowered):
            return [lowered]

        # 설명형 문장은 토큰으로 쪼개서 핵심 명사만 남긴다.
        tokens = [
            token
            for token in re.findall(r"[a-z0-9가-힣]+", lowered)
            if token not in _CAPTION_STOPWORDS and len(token) >= 3
        ]
        return tokens[:4]

    def _looks_like_object_phrase(self, chunk: str) -> bool:
        if any(chunk.startswith(prefix) for prefix in _CAPTION_STOP_PREFIXES):
            return False
        if len(chunk.split()) > 3:
            return False
        return True


def _run_florence_prompt(
    *,
    model: object,
    processor: object,
    task_prompt: str,
    asset_path: Path,
    device: str,
    torch_dtype: str,
    max_new_tokens: int,
    num_beams: int,
) -> object:
    import torch

    image = _load_image_as_rgb(asset_path)
    inputs = processor(text=task_prompt, images=image, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs["pixel_values"].to(device, getattr(torch, torch_dtype))

    generated_ids = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=num_beams,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height),
    )


def _load_image_as_rgb(asset_path: Path):
    from PIL import Image

    if asset_path.suffix.lower() == ".svg":
        try:
            import cairosvg
        except ImportError as exc:
            raise TaggingError(
                "SVG tagging requires cairosvg to rasterize vector images."
            ) from exc

        # 벡터 입력도 모델이 이해할 수 있도록 런타임에 RGB raster로 맞춘다.
        png_bytes = cairosvg.svg2png(bytestring=asset_path.read_bytes())
        return Image.open(BytesIO(png_bytes)).convert("RGB")

    return Image.open(asset_path).convert("RGB")
