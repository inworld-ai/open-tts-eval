from __future__ import annotations

import json

from tts_assess.sampling.providers import _http
from tts_assess.sampling.providers.base import (
    ProviderError,
    SynthesisRequest,
    SynthesisResult,
    TTSProvider,
    Voice,
    redact_text,
)


class ElevenLabsProvider(TTSProvider):
    """ElevenLabs TTS. Docs: https://elevenlabs.io/docs/api-reference/text-to-speech"""

    name = "elevenlabs"
    default_model = "eleven_multilingual_v2"
    forced_encoding = "WAV"
    base_url = "https://api.elevenlabs.io/v1"
    output_format = "wav_44100"
    # ``voice_settings.speed`` is the only pacing control; there is no temperature.
    supported_controls = frozenset({"speaking_rate"})
    # Models that honour ``language_code``. For any other model the API ignores
    # it, so it is not sent (and therefore not recorded as applied).
    language_enforcing_models = frozenset({"eleven_turbo_v2_5", "eleven_flash_v2_5"})

    def __init__(
        self, api_key: str, *, timeout: float = 120.0, max_retries: int = 3, transport=None
    ):
        if not api_key:
            raise ValueError("ElevenLabs API key is required")
        self._key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._transport = transport or _http.urllib_transport

    def _headers(self) -> dict[str, str]:
        return {"xi-api-key": self._key, "Content-Type": "application/json"}

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        self.check_request(request)
        query = f"?output_format={self.output_format}"
        url = f"{self.base_url}/text-to-speech/{request.voice_id}{query}"
        payload: dict[str, object] = {"text": request.text, "model_id": request.model_id}
        if request.speaking_rate is not None:
            payload["voice_settings"] = {"speed": request.speaking_rate}
        language = _iso639_1(request.language)
        if language and request.model_id in self.language_enforcing_models:
            payload["language_code"] = language
        status, data = _http.call(
            self._transport,
            "POST",
            url,
            self._headers(),
            body=json.dumps(payload).encode("utf-8"),
            timeout=self._timeout,
            max_retries=self._max_retries,
        )
        if status != 200:
            raise ProviderError(f"ElevenLabs TTS HTTP {status}: {_http.snippet(data)}")
        return SynthesisResult(
            audio=data,
            audio_encoding="WAV",
            sample_rate_hz=44100,
            usage={},
            request_sent={**redact_text(payload), "output_format": self.output_format},
        )

    def list_voices(self) -> list[Voice]:
        status, data = _http.call(
            self._transport, "GET", f"{self.base_url}/voices", self._headers(),
            timeout=self._timeout,
        )
        if status != 200:
            raise ProviderError(f"ElevenLabs voices HTTP {status}: {_http.snippet(data)}")
        voices = []
        for entry in json.loads(data).get("voices", []):
            voice_id = entry.get("voice_id")
            if not voice_id:
                continue
            labels = entry.get("labels", {}) or {}
            lang = labels.get("language") or labels.get("accent")
            voices.append(
                Voice(
                    voice_id=voice_id,
                    name=entry.get("name"),
                    languages=(lang,) if lang else (),
                    metadata=labels,
                )
            )
        return voices


def _iso639_1(language: str | None) -> str | None:
    """'en-US' -> 'en'; ElevenLabs takes a two-letter code."""
    if not language:
        return None
    code = language.split("-")[0].split("_")[0].strip().lower()
    return code or None
