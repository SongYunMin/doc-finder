from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import re

from doc_finder.models.florence_2.tagger import _load_image_as_rgb
from doc_finder.services.query_normalizer import QueryNormalizer
from doc_finder.services.tagging_service import TaggingError, TaggingResult, clean_tag_candidates

_DESCRIPTION_STOP_PREFIXES = (
    "the image",
    "this image",
    "the photo",
    "photo of",
    "the scene",
    "showing",
    "shows",
    "there is",
    "there are",
)

_DESCRIPTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "image",
    "in",
    "is",
    "of",
    "on",
    "the",
    "this",
    "with",
}


class PaliGemma2VisionTagger:
    """PaliGemma 2 mix 체크포인트를 검색용 태그 계약으로 감싸는 어댑터."""

    def __init__(
        self,
        model_id: str,
        device: str,
        torch_dtype: str,
        query_normalizer: QueryNormalizer,
        max_new_tokens: int = 128,
        prompt_runner: Callable[[str, Path], object] | None = None,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._torch_dtype = torch_dtype
        self._query_normalizer = query_normalizer
        self._max_new_tokens = max_new_tokens
        self._prompt_runner = prompt_runner
        self._model = None
        self._processor = None

    def tag(self, asset_path: Path, sha256: str) -> TaggingResult:
        del sha256

        # v1은 설명형 프롬프트와 OCR 프롬프트를 합쳐 검색용 태그를 만든다.
        try:
            describe_result = self._run_prompt("describe en", asset_path)
            ocr_result = self._run_prompt("ocr", asset_path)
        except Exception as exc:  # noqa: BLE001
            raise TaggingError(self._build_runtime_error_message(asset_path, exc)) from exc

        describe_tags = self._extract_describe_tags(describe_result)
        ocr_tags = self._extract_ocr_tags(ocr_result)
        keyword_tags = clean_tag_candidates([*describe_tags, *ocr_tags], max_tags=12)
        normalized_tags = self._build_normalized_tags(keyword_tags)
        confidence = self._compute_confidence(describe_tags, ocr_tags, normalized_tags)
        review_status = "approved" if confidence >= 0.80 else "pending"

        return TaggingResult(
            keyword_tags=keyword_tags,
            normalized_tags=normalized_tags,
            confidence=confidence,
            review_status=review_status,
        )

    def _run_prompt(self, task_prompt: str, asset_path: Path) -> str:
        if self._prompt_runner is not None:
            return str(self._prompt_runner(task_prompt, asset_path))

        model, processor = self._get_model_bundle()
        return _run_paligemma_prompt(
            model=model,
            processor=processor,
            task_prompt=task_prompt,
            asset_path=asset_path,
            device=self._device,
            torch_dtype=self._torch_dtype,
            max_new_tokens=self._max_new_tokens,
        )

    def _get_model_bundle(self) -> tuple[object, object]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor

        import torch
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        dtype = getattr(torch, self._torch_dtype)
        self._model = PaliGemmaForConditionalGeneration.from_pretrained(
            self._model_id,
            torch_dtype=dtype,
        ).to(self._device)
        self._processor = AutoProcessor.from_pretrained(self._model_id)
        return self._model, self._processor

    def _extract_describe_tags(self, payload: object) -> list[str]:
        description_text = str(payload or "")
        raw_candidates: list[str] = []
        for chunk in re.split(r"[,;\n\.]+", description_text):
            cleaned = chunk.strip()
            if cleaned:
                raw_candidates.extend(self._normalize_description_chunk(cleaned))
        return clean_tag_candidates(raw_candidates)

    def _extract_ocr_tags(self, payload: object) -> list[str]:
        ocr_text = str(payload or "")
        candidates = [
            token
            for token in re.findall(r"[A-Za-z0-9가-힣]+", ocr_text)
            if token.strip()
        ]
        # OCR에서는 한 글자 라벨과 짧은 숫자를 버리면 검색 회복력이 크게 떨어진다.
        return clean_tag_candidates(candidates)

    def _build_normalized_tags(self, keyword_tags: list[str]) -> list[str]:
        normalized_tags: list[str] = []
        for tag in keyword_tags:
            normalized_tags.extend(self._query_normalizer.normalize_tag_candidates(tag))
        return clean_tag_candidates(normalized_tags)

    def _compute_confidence(
        self,
        describe_tags: list[str],
        ocr_tags: list[str],
        normalized_tags: list[str],
    ) -> float:
        # Florence와 같은 0.95 ceiling을 유지하되, describe/ocr 각각의 근거를 반영한다.
        confidence = 0.45
        if describe_tags:
            confidence += 0.25
        if ocr_tags:
            confidence += 0.15
        if normalized_tags and normalized_tags != clean_tag_candidates(describe_tags):
            confidence += 0.10
        return round(min(confidence, 0.95), 2)

    def _normalize_description_chunk(self, chunk: str) -> list[str]:
        lowered = " ".join(chunk.lower().split())
        if not lowered:
            return []

        if self._looks_like_object_phrase(lowered):
            return [lowered]

        # 설명형 문장은 핵심 명사 토큰으로 줄여 noise를 줄인다.
        tokens = [
            token
            for token in re.findall(r"[a-z0-9가-힣]+", lowered)
            if token not in _DESCRIPTION_STOPWORDS and len(token) >= 3
        ]
        return tokens[:4]

    def _looks_like_object_phrase(self, chunk: str) -> bool:
        if any(chunk.startswith(prefix) for prefix in _DESCRIPTION_STOP_PREFIXES):
            return False
        if len(chunk.split()) > 3:
            return False
        return True

    def _build_runtime_error_message(self, asset_path: Path, exc: Exception) -> str:
        error_text = str(exc)
        lowered_error_text = error_text.casefold()

        # 기본 gated repo 실패는 런타임 버그가 아니라 권한 문제이므로 바로 조치 문구를 준다.
        if "gated repo" in lowered_error_text or "restricted" in lowered_error_text:
            return (
                f"PaliGemma 2 model access failed for {asset_path.name}. "
                f"Request access to {self._model_id} on Hugging Face and authenticate with "
                "`HF_TOKEN` before retrying."
            )

        return f"PaliGemma 2 tagging failed for {asset_path.name}."


def _run_paligemma_prompt(
    *,
    model: object,
    processor: object,
    task_prompt: str,
    asset_path: Path,
    device: str,
    torch_dtype: str,
    max_new_tokens: int,
) -> str:
    import torch

    image = _load_image_as_rgb(asset_path)
    inputs = processor(images=image, text=task_prompt, return_tensors="pt")
    prepared_inputs = {}
    for key, value in inputs.items():
        if key == "pixel_values":
            prepared_inputs[key] = value.to(device, getattr(torch, torch_dtype))
        else:
            prepared_inputs[key] = value.to(device)

    generated_ids = model.generate(
        **prepared_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    input_length = prepared_inputs["input_ids"].shape[-1]
    generated_tokens = generated_ids[:, input_length:]
    return processor.decode(generated_tokens[0], skip_special_tokens=True).strip()
