from __future__ import annotations

from tts_assess.sampling.providers.base import (
    ProviderError,
    SynthesisRequest,
    SynthesisResult,
    TTSProvider,
    Voice,
    file_extension,
)
from tts_assess.sampling.providers.elevenlabs import ElevenLabsProvider
from tts_assess.sampling.providers.gradium import GradiumProvider
from tts_assess.sampling.providers.hume import HumeProvider
from tts_assess.sampling.providers.inworld import InworldProvider

_PROVIDERS: dict[str, type[TTSProvider]] = {
    provider.name: provider
    for provider in (
        InworldProvider,
        ElevenLabsProvider,
        HumeProvider,
        GradiumProvider,
    )
}


def available_providers() -> list[str]:
    return sorted(_PROVIDERS)


def build_provider(name: str, api_key: str, **kwargs) -> TTSProvider:
    try:
        provider_cls = _PROVIDERS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown provider {name!r}; available: {', '.join(available_providers())}"
        ) from exc
    return provider_cls(api_key=api_key, **kwargs)


__all__ = [
    "ElevenLabsProvider",
    "GradiumProvider",
    "HumeProvider",
    "InworldProvider",
    "ProviderError",
    "SynthesisRequest",
    "SynthesisResult",
    "TTSProvider",
    "Voice",
    "available_providers",
    "build_provider",
    "file_extension",
]
