from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable

from tts_assess.sampling.providers.base import (
    ProviderError,
    SynthesisRequest,
    SynthesisResult,
    TTSProvider,
    Voice,
    redact_text,
)

# transport(method, url, headers, body, timeout) -> (status_code, response_bytes).
# Injectable so the provider can be tested without real network access.
Transport = Callable[[str, str, dict[str, str], bytes | None, float], tuple[int, bytes]]

_RETRY_STATUS = {429, 500, 502, 503, 504}


class InworldProvider(TTSProvider):
    """Inworld.ai Realtime TTS REST backend.

    Docs: https://docs.inworld.ai/tts/tts — POST /tts/v1/voice with Basic auth,
    returns base64 audio in ``audioContent``.
    """

    name = "inworld"
    default_model = "inworld-tts-1.5-max"
    synth_url = "https://api.inworld.ai/tts/v1/voice"
    voices_url = "https://api.inworld.ai/voices/v1/voices"
    supported_controls = frozenset({"speaking_rate", "temperature"})

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 60.0,
        max_retries: int = 3,
        transport: Transport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Inworld API key is required (set INWORLD_API_KEY)")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max(1, max_retries)
        self._transport = transport or _urllib_transport

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        self.check_request(request)
        audio_config: dict[str, object] = {
            "audioEncoding": request.audio_encoding,
            "sampleRateHertz": request.sample_rate_hz,
        }
        if request.speaking_rate is not None:
            audio_config["speakingRate"] = request.speaking_rate
        payload: dict[str, object] = {
            "text": request.text,
            "voiceId": request.voice_id,
            "modelId": request.model_id,
            "audioConfig": audio_config,
        }
        if request.language:
            payload["language"] = request.language
        if request.temperature is not None:
            payload["temperature"] = request.temperature

        data = self._call("POST", self.synth_url, payload)
        audio_content = data.get("audioContent")
        if not audio_content:
            raise ProviderError(f"Inworld response missing audioContent (keys: {sorted(data)})")
        usage = data.get("usage", {}) or {}
        return SynthesisResult(
            audio=base64.b64decode(audio_content),
            audio_encoding=request.audio_encoding,
            sample_rate_hz=request.sample_rate_hz,
            usage=usage,
            # Inworld reports the model that actually served the request; it can
            # differ from the requested id when a model is routed or retired.
            model_id=usage.get("modelId") or None,
            request_sent=redact_text(payload),
        )

    def list_voices(self) -> list[Voice]:
        data = self._call("GET", self.voices_url)
        voices: list[Voice] = []
        for item in data.get("voices", []) or []:
            voice_id = (
                item.get("voiceId")
                or item.get("voice_id")
                or item.get("id")
                or item.get("name")
            )
            if not voice_id:
                continue
            languages = item.get("languages") or item.get("language") or []
            if isinstance(languages, str):
                languages = [languages]
            voices.append(
                Voice(
                    voice_id=voice_id,
                    name=item.get("displayName") or item.get("name"),
                    languages=tuple(languages),
                    metadata=item,
                )
            )
        return voices

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Basic {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _call(self, method: str, url: str, payload: dict | None = None) -> dict:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        for attempt in range(self._max_retries):
            status, data = self._transport(method, url, self._headers(), body, self._timeout)
            if status == 200:
                return json.loads(data.decode("utf-8"))
            if status in _RETRY_STATUS and attempt < self._max_retries - 1:
                time.sleep(min(2**attempt, 8))
                continue
            raise ProviderError(
                f"Inworld API {method} {url} failed with HTTP {status}: {_snippet(data)}"
            )
        raise ProviderError(f"Inworld API {method} {url} exhausted retries")


def _urllib_transport(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout: float,
) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        raise ProviderError(f"network error calling {url}: {exc.reason}") from exc


def _snippet(data: bytes, limit: int = 300) -> str:
    text = data.decode("utf-8", errors="replace").strip()
    return text if len(text) <= limit else text[:limit] + "…"
