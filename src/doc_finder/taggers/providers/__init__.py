"""기본 제공 태거 provider 모듈 등록."""

from doc_finder.taggers.providers import (  # noqa: E402
    http,
    static,
)  # noqa: F401

__all__ = [
    "http",
    "static",
]
