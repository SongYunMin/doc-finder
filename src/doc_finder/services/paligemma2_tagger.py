"""기존 서비스 경로와의 호환을 위한 PaliGemma 2 shim 모듈."""

from doc_finder.models.paligemma_2 import PaliGemma2VisionTagger, _load_image_as_rgb

__all__ = ["PaliGemma2VisionTagger", "_load_image_as_rgb"]
