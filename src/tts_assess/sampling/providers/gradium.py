from __future__ import annotations

import io
import json
import math
import wave

from tts_assess.sampling.providers import _http
from tts_assess.sampling.providers.base import (
    ProviderError,
    SynthesisRequest,
    SynthesisResult,
    TTSProvider,
    Voice,
    redact_text,
)


class GradiumProvider(TTSProvider):
    """Gradium REST TTS. Docs: https://docs.gradium.ai/guides/text-to-speech-rest"""

    name = "gradium"
    default_model = "default"
    forced_encoding = "WAV"
    base_url = "https://api.gradium.ai"

    def __init__(
        self, api_key: str, *, timeout: float = 120.0, max_retries: int = 3, transport=None
    ):
        if not api_key:
            raise ValueError("Gradium API key is required")
        self._key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._transport = transport or _http.urllib_transport

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._key, "Content-Type": "application/json"}

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        if request.speaking_rate is not None:
            raise ProviderError("Gradium does not support speaking_rate")
        payload = {
            "text": request.text,
            "voice_id": request.voice_id,
            "model_name": request.model_id or self.default_model,
            "output_format": "wav",
            "only_audio": True,
        }
        if request.temperature is not None:
            if not math.isfinite(request.temperature) or not 0 <= request.temperature <= 1.5:
                raise ProviderError("Gradium temperature must be finite and between 0 and 1.5")
            payload["json_config"] = json.dumps({"temp": request.temperature})
        status, data = _http.call(
            self._transport,
            "POST",
            f"{self.base_url}/api/post/speech/tts",
            self._headers(),
            body=json.dumps(payload).encode("utf-8"),
            timeout=self._timeout,
            max_retries=self._max_retries,
        )
        if status != 200:
            raise ProviderError(f"Gradium synthesize failed: HTTP {status}: {_http.snippet(data)}")
        try:
            with wave.open(io.BytesIO(data), "rb") as audio:
                sample_rate = audio.getframerate()
                frames = audio.getnframes()
                frame_size = audio.getnchannels() * audio.getsampwidth()
                if sample_rate <= 0 or frames <= 0 or frame_size <= 0:
                    raise ValueError("empty audio or invalid sample rate/frame size")
                pcm = audio.readframes(frames)
                # Streaming WAVs can use 0xFFFFFFFF for an unknown data length.
                unknown_length = frames == 0xFFFFFFFF // frame_size
                if not pcm or len(pcm) % frame_size:
                    raise ValueError("empty audio or incomplete frame")
                if not unknown_length and len(pcm) != frames * frame_size:
                    raise ValueError("truncated audio")
                if unknown_length:
                    # Rewrite the header with the real length so the saved file is a
                    # well-formed WAV for tools that trust RIFF sizes.
                    fixed = io.BytesIO()
                    with wave.open(fixed, "wb") as out:
                        out.setnchannels(audio.getnchannels())
                        out.setsampwidth(audio.getsampwidth())
                        out.setframerate(sample_rate)
                        out.writeframes(pcm)
                    data = fixed.getvalue()
        except (wave.Error, EOFError, ValueError, RuntimeError) as exc:
            raise ProviderError(f"Gradium returned invalid WAV audio: {exc}") from exc
        return SynthesisResult(
            audio=data,
            audio_encoding="WAV",
            sample_rate_hz=sample_rate,
            request_sent=redact_text(payload),
        )

    def list_voices(self) -> list[Voice]:
        status, data = _http.call(
            self._transport,
            "GET",
            f"{self.base_url}/api/voices/?include_catalog=true&limit=0",
            self._headers(),
            timeout=self._timeout,
            max_retries=self._max_retries,
        )
        if status != 200:
            raise ProviderError(f"Gradium voices HTTP {status}: {_http.snippet(data)}")
        try:
            entries = json.loads(data)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ProviderError("Gradium returned invalid voices JSON") from exc
        if not isinstance(entries, list):
            raise ProviderError("Gradium voices response must be an array")
        voices = []
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("uid"), str)
                or not entry["uid"]
                or not isinstance(entry.get("is_catalog"), bool)
                or not isinstance(entry.get("name"), str)
                or (entry.get("language") is not None and not isinstance(entry["language"], str))
            ):
                raise ProviderError("Gradium returned an invalid voice entry")
            if entry["is_catalog"]:
                language = entry.get("language")
                voices.append(
                    Voice(
                        voice_id=entry["uid"],
                        name=entry["name"],
                        languages=(language,) if language else (),
                        metadata=entry,
                    )
                )
        return voices
