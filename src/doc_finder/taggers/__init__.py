"""태거 provider registry 공개 진입점."""

from doc_finder.taggers.registry import (
    available_tagger_providers,
    build_tagger,
    build_tagger_from_env,
    register_tagger_provider,
)

__all__ = [
    "available_tagger_providers",
    "build_tagger",
    "build_tagger_from_env",
    "register_tagger_provider",
]
