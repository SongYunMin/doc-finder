"""기본 제공 태거 provider 모듈 등록."""

from doc_finder.taggers.providers import (
    florence2,
    florence2_large_ft,
    http,
    paligemma2,
    static,
)  # noqa: F401

__all__ = ["florence2", "florence2_large_ft", "http", "paligemma2", "static"]
