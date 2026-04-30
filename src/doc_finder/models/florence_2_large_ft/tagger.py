from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from doc_finder.models.florence_2.tagger import Florence2VisionTagger, _load_image_as_rgb
from doc_finder.services.query_normalizer import QueryNormalizer


class Florence2LargeFtVisionTagger(Florence2VisionTagger):
    """`microsoft/Florence-2-large-ft` 기본값을 쓰는 Florence-2 preview 태거."""

    DEFAULT_MODEL_ID = "microsoft/Florence-2-large-ft"

    def __init__(
        self,
        *,
        device: str,
        torch_dtype: str,
        query_normalizer: QueryNormalizer,
        model_id: str | None = None,
        max_new_tokens: int = 512,
        num_beams: int = 3,
        prompt_runner: Callable[[str, Path], object] | None = None,
    ) -> None:
        # CLI에서 base large와 large-ft를 같은 provider 경로로 비교할 수 있게 모델 id override를 허용한다.
        super().__init__(
            model_id=model_id or self.DEFAULT_MODEL_ID,
            device=device,
            torch_dtype=torch_dtype,
            query_normalizer=query_normalizer,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            prompt_runner=prompt_runner,
        )
