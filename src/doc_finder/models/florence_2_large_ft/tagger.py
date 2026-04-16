from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from doc_finder.models.florence_2.tagger import Florence2VisionTagger, _load_image_as_rgb
from doc_finder.services.query_normalizer import QueryNormalizer


class Florence2LargeFtVisionTagger(Florence2VisionTagger):
    """`microsoft/Florence-2-large-ft` 전용 런타임 태거."""

    DEFAULT_MODEL_ID = "microsoft/Florence-2-large-ft"

    def __init__(
        self,
        *,
        device: str,
        torch_dtype: str,
        query_normalizer: QueryNormalizer,
        max_new_tokens: int = 512,
        num_beams: int = 3,
        prompt_runner: Callable[[str, Path], object] | None = None,
    ) -> None:
        # large-ft는 모델 id를 고정해 별도 런타임 네임스페이스로 분리한다.
        super().__init__(
            model_id=self.DEFAULT_MODEL_ID,
            device=device,
            torch_dtype=torch_dtype,
            query_normalizer=query_normalizer,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            prompt_runner=prompt_runner,
        )
