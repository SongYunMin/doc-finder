"""기존 서비스 경로와의 호환을 위한 Florence-2 shim 모듈."""

from doc_finder.models.florence_2 import Florence2VisionTagger, _load_image_as_rgb

__all__ = ["Florence2VisionTagger", "_load_image_as_rgb"]
