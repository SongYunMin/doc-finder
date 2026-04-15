from __future__ import annotations

from collections.abc import Callable, Mapping
import os

from doc_finder.services.query_normalizer import QueryNormalizer
from doc_finder.services.tagging_service import VisionTagger

TaggerProviderBuilder = Callable[..., VisionTagger]

_PROVIDER_BUILDERS: dict[str, TaggerProviderBuilder] = {}
_BUILTIN_PROVIDERS_LOADED = False


def register_tagger_provider(name: str, builder: TaggerProviderBuilder) -> None:
    normalized_name = name.casefold().strip()
    if not normalized_name:
        raise ValueError("Tagger provider name must not be empty.")
    _PROVIDER_BUILDERS[normalized_name] = builder


def available_tagger_providers() -> tuple[str, ...]:
    _ensure_builtin_providers_registered()
    return tuple(sorted(_PROVIDER_BUILDERS))


def build_tagger(
    provider: str,
    *,
    query_normalizer: QueryNormalizer | None = None,
    environ: Mapping[str, str] | None = None,
) -> VisionTagger:
    _ensure_builtin_providers_registered()
    normalized_provider = provider.casefold().strip()
    try:
        builder = _PROVIDER_BUILDERS[normalized_provider]
    except KeyError as exc:
        available = ", ".join(sorted(_PROVIDER_BUILDERS)) or "<none>"
        raise ValueError(
            f"Unsupported tagger provider: {provider}. Available providers: {available}"
        ) from exc

    runtime_environ = dict(os.environ if environ is None else environ)
    return builder(
        query_normalizer=query_normalizer or QueryNormalizer(),
        environ=runtime_environ,
    )


def build_tagger_from_env(
    *,
    query_normalizer: QueryNormalizer | None = None,
    environ: Mapping[str, str] | None = None,
) -> VisionTagger:
    runtime_environ = dict(os.environ if environ is None else environ)
    provider = runtime_environ.get("DOC_FINDER_TAGGER_PROVIDER", "http")
    return build_tagger(
        provider,
        query_normalizer=query_normalizer,
        environ=runtime_environ,
    )


def _ensure_builtin_providers_registered() -> None:
    global _BUILTIN_PROVIDERS_LOADED
    if _BUILTIN_PROVIDERS_LOADED:
        return

    # provider 모듈 import 시 register_tagger_provider 사이드이펙트로 기본 registry를 채운다.
    from doc_finder.taggers import providers  # noqa: F401

    _BUILTIN_PROVIDERS_LOADED = True
