"""기본 제공 태거 provider 모듈 등록."""


def _detect_torch_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


from doc_finder.taggers.providers import (  # noqa: E402
    florence2,
    florence2_large_ft,
    http,
    static,
)  # noqa: F401

__all__ = [
    "florence2",
    "florence2_large_ft",
    "http",
    "static",
]
