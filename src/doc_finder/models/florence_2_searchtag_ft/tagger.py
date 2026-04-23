from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import re

from doc_finder.models.florence_2.tagger import _load_image_as_rgb
from doc_finder.services.query_normalizer import QueryNormalizer
from doc_finder.services.tagging_service import (
    TaggingError,
    TaggingResult,
    build_normalized_tags,
    clean_tag_candidates,
    compute_confidence,
)


class Florence2SearchTagVisionTagger:
    """`<SearchTag>` 프롬프트 하나로 canonical 태그 문자열을 생성하는 Florence 런타임."""

    TASK_PROMPT = "<SearchTag>"
    OCR_PROMPT = "<OCR>"

    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        torch_dtype: str,
        query_normalizer: QueryNormalizer,
        max_new_tokens: int = 128,
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
        del sha256

        try:
            response = self._run_prompt(self.TASK_PROMPT, asset_path)
            ocr_response = self._run_prompt(self.OCR_PROMPT, asset_path)
        except Exception as exc:  # noqa: BLE001
            raise TaggingError(
                f"Florence-2 SearchTag tagging failed for {asset_path.name}."
            ) from exc

        keyword_tags = clean_tag_candidates(
            [
                *self._extract_keyword_tags(response),
                *self._extract_ocr_tags(ocr_response),
            ],
            max_tags=12,
        )
        normalized_tags = build_normalized_tags(keyword_tags, self._query_normalizer)
        confidence = compute_confidence(
            primary_signal=bool(self._extract_keyword_tags(response)),
            secondary_signal=bool(self._extract_ocr_tags(ocr_response)),
            expansion_signal=bool(normalized_tags),
        )
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
        return _run_searchtag_prompt(
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

    def _extract_keyword_tags(self, payload: object) -> list[str]:
        if isinstance(payload, dict):
            response_text = str(payload.get(self.TASK_PROMPT, ""))
        else:
            response_text = str(payload or "")

        # fine-tuned 경로는 이미 세미콜론 구분 태그 문자열을 내놓는 것을 전제로 한다.
        candidates = [
            chunk.strip()
            for chunk in re.split(r"[;\n,]+", response_text)
            if chunk.strip()
        ]
        return clean_tag_candidates(candidates, max_tags=12)

    def _extract_ocr_tags(self, payload: object) -> list[str]:
        if isinstance(payload, dict):
            response_text = str(payload.get(self.OCR_PROMPT, ""))
        else:
            response_text = str(payload or "")

        candidates = [
            self._to_text_tag(chunk)
            for chunk in re.split(r"[\n;]+", response_text)
            if chunk.strip()
        ]
        return clean_tag_candidates([candidate for candidate in candidates if candidate])

    @staticmethod
    def _to_text_tag(chunk: str) -> str:
        cleaned_chunk = " ".join(str(chunk).split()).strip()
        if not cleaned_chunk:
            return ""
        return f"text:{cleaned_chunk}"


def _run_searchtag_prompt(
    *,
    model: object,
    processor: object,
    task_prompt: str,
    asset_path: Path,
    device: str,
    torch_dtype: str,
    max_new_tokens: int,
    num_beams: int,
) -> str:
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
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    if generated_text.casefold().startswith(task_prompt.casefold()):
        generated_text = generated_text[len(task_prompt) :].strip(" :;\n\t")
    return generated_text
