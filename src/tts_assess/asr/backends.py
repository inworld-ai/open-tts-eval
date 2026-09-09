from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from tts_assess.audio import resample_mono
from tts_assess.config import ASRConfig

# faster-whisper only accepts decoded arrays at this rate.
WHISPER_SAMPLE_RATE = 16000
# faster-whisper retries a degenerate segment (repetition, low log-prob) with
# temperature sampling. Seeding the decoder before every clip makes that
# fallback reproducible per clip and independent of the clips before it.
ASR_DECODER_SEED = 0


@dataclass(frozen=True)
class ASRSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class ASRTranscript:
    text: str
    segments: list[ASRSegment]
    backend: str
    model: str


def transcribe(
    source: Path | np.ndarray,
    config: ASRConfig,
    *,
    expected_text: str | None = None,
    sample_rate: int | None = None,
) -> ASRTranscript:
    """Transcribe an audio file path or an already-decoded mono float32 array.

    Array input requires ``sample_rate``; it is what keeps the CLI pipeline and
    in-memory callers (services) on the identical code path.
    """
    if config.backend == "mock":
        text = expected_text or ""
        return ASRTranscript(
            text=text,
            segments=[ASRSegment(start=0.0, end=0.0, text=text)] if text else [],
            backend="mock",
            model="mock",
        )
    return _transcribe_faster_whisper(source, config, sample_rate=sample_rate)


def _seed_decoder(seed: int) -> None:
    """Seed CTranslate2's RNG (used by faster-whisper's temperature fallback).

    Removes the sampling randomness from transcripts. It does not make CPU
    inference bit-exact: float reduction order can still flip a near-tie beam
    decision on a handful of clips per thousand.
    """
    try:
        import ctranslate2
    except ImportError:
        return
    ctranslate2.set_random_seed(seed)


@lru_cache(maxsize=4)
def _load_whisper_model(model: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Install with `pip install .[asr]` or use "
            "`asr.backend: mock` for tests."
        ) from exc
    return WhisperModel(model, device=device, compute_type=compute_type)


def _transcribe_faster_whisper(
    source: Path | np.ndarray,
    config: ASRConfig,
    *,
    sample_rate: int | None = None,
) -> ASRTranscript:
    if isinstance(source, np.ndarray):
        if sample_rate is None:
            raise ValueError("sample_rate is required when transcribing a decoded array")
        audio_input = resample_mono(
            np.asarray(source, dtype=np.float32), sample_rate, WHISPER_SAMPLE_RATE
        )
    else:
        audio_input = str(source)
    # Cached so the model loads once per (model, device, compute_type) instead of
    # being re-instantiated for every manifest row.
    model = _load_whisper_model(config.model, config.device, config.compute_type)
    _seed_decoder(ASR_DECODER_SEED)
    segments_iter, _info = model.transcribe(
        audio_input,
        language=config.language,
        beam_size=5,
        vad_filter=True,
    )
    segments = [
        ASRSegment(start=float(segment.start), end=float(segment.end), text=segment.text.strip())
        for segment in segments_iter
    ]
    return ASRTranscript(
        text=" ".join(segment.text for segment in segments).strip(),
        segments=segments,
        backend="faster-whisper",
        model=config.model,
    )
