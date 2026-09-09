from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


@dataclass(frozen=True)
class AudioFeatures:
    sample_rate: int
    channels: int
    duration_sec: float
    rms: float
    peak: float
    clipping_ratio: float
    leading_silence_sec: float
    trailing_silence_sec: float
    silence_ratio: float
    tail_click_detected: bool
    tail_click_score: float


def read_audio(path: Path) -> tuple[np.ndarray, int, int]:
    """Decode a file once into (mono float32, sample_rate, channels).

    Raises ValueError for NaN/Inf samples (possible in FLOAT WAVs): a single NaN
    poisons every comparison downstream and would otherwise pass as "silent".
    """
    raw, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    channels = raw.shape[1]
    mono = raw.mean(axis=1)
    _require_finite(mono)
    return mono, sample_rate, channels


def read_audio_bytes(data: bytes) -> tuple[np.ndarray, int, int]:
    """Decode an in-memory audio container into (mono float32, sample_rate, channels)."""
    raw, sample_rate = sf.read(io.BytesIO(data), always_2d=True, dtype="float32")
    channels = raw.shape[1]
    mono = raw.mean(axis=1)
    _require_finite(mono)
    return mono, sample_rate, channels


def _require_finite(audio: np.ndarray) -> None:
    if len(audio) and not np.isfinite(audio).all():
        bad = int(np.count_nonzero(~np.isfinite(audio)))
        raise ValueError(f"audio contains {bad} non-finite sample(s) (NaN/Inf)")


def resample_mono(audio: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if not target_sample_rate or sample_rate == target_sample_rate:
        return audio
    gcd = np.gcd(sample_rate, target_sample_rate)
    return signal.resample_poly(
        audio, target_sample_rate // gcd, sample_rate // gcd
    ).astype(np.float32)


def load_audio(path: Path, target_sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    mono, sample_rate, _ = read_audio(path)
    if target_sample_rate and sample_rate != target_sample_rate:
        mono = resample_mono(mono, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    return mono, sample_rate


def analyze_audio(path: Path) -> AudioFeatures:
    audio, sample_rate, channels = read_audio(path)
    return analyze_features(audio, sample_rate, channels)


def analyze_features(audio: np.ndarray, sample_rate: int, channels: int) -> AudioFeatures:
    _require_finite(audio)
    duration = float(len(audio) / sample_rate) if sample_rate else 0.0
    abs_audio = np.abs(audio)
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    peak = float(abs_audio.max()) if len(audio) else 0.0
    clipping_ratio = float(np.mean(abs_audio >= 0.999)) if len(audio) else 0.0
    silence_mask = _silence_mask(audio)
    leading = _edge_silence_seconds(silence_mask, sample_rate, from_start=True)
    trailing = _edge_silence_seconds(silence_mask, sample_rate, from_start=False)
    silence_ratio = float(np.mean(silence_mask)) if len(silence_mask) else 1.0
    click_score = _tail_click_score(audio, sample_rate)
    return AudioFeatures(
        sample_rate=sample_rate,
        channels=channels,
        duration_sec=duration,
        rms=rms,
        peak=peak,
        clipping_ratio=clipping_ratio,
        leading_silence_sec=leading,
        trailing_silence_sec=trailing,
        silence_ratio=silence_ratio,
        tail_click_detected=click_score >= TAIL_CLICK_THRESHOLD,
        tail_click_score=click_score,
    )


def chars_per_second(text: str, duration_sec: float) -> float | None:
    if duration_sec <= 0:
        return None
    return len(text) / duration_sec


# Frames below this RMS (about -80 dBFS) are digital silence and never count as
# speech. Leading/trailing runs of them are padding: they are excluded when the
# noise floor and speech level are estimated, so appending silence to a clip can
# never lower the threshold and re-label its existing quiet frames as speech.
SILENCE_FLOOR = 1e-4
# Noise-floor and speech-level factors of the adaptive threshold (see _silence_mask).
NOISE_FLOOR_FACTOR = 1.5
SPEECH_FLOOR_FACTOR = 0.1
# Tail clicks are scored on audio resampled to this rate. The score is a
# sample-to-sample difference, which for the same analog waveform scales with the
# sample rate; a fixed analysis rate keeps the threshold meaningful across
# providers that deliver 24 kHz vs 44.1/48 kHz audio.
CLICK_ANALYSIS_RATE = 24000
TAIL_CLICK_THRESHOLD = 4.0


def _silence_mask(audio: np.ndarray, frame_size: int = 1024) -> np.ndarray:
    """Per-sample silence mask from an adaptive frame-energy threshold.

    A frame is silent when its RMS is at or below
    ``max(min(1.5 * p20, 0.1 * p95), SILENCE_FLOOR)``, where the percentiles are
    taken over the clip's content between its first and last non-silent frame.
    Internal pauses shape the noise floor as before; leading/trailing digital
    silence is padding and does not, which keeps the verdict stable when a
    clip is padded (see tests/test_metric_robustness.py).
    """
    if len(audio) == 0:
        return np.array([], dtype=bool)
    pad = (-len(audio)) % frame_size
    padded = np.pad(audio, (0, pad)) if pad else audio
    frames = padded.reshape(-1, frame_size)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    active = np.flatnonzero(frame_rms > SILENCE_FLOOR)
    if len(active):
        core = frame_rms[active[0] : active[-1] + 1]
        noise_floor = float(np.percentile(core, 20)) * NOISE_FLOOR_FACTOR
        speech_floor = float(np.percentile(core, 95)) * SPEECH_FLOOR_FACTOR
        threshold = max(min(noise_floor, speech_floor), SILENCE_FLOOR)
    else:
        threshold = SILENCE_FLOOR
    frame_silence = frame_rms <= threshold
    return np.repeat(frame_silence, frame_size)[: len(audio)]


def _edge_silence_seconds(mask: np.ndarray, sample_rate: int, *, from_start: bool) -> float:
    if len(mask) == 0:
        return 0.0
    iterable = mask if from_start else mask[::-1]
    count = 0
    for silent in iterable:
        if not silent:
            break
        count += 1
    return count / sample_rate


def _tail_click_score(audio: np.ndarray, sample_rate: int) -> float:
    """Largest sample step in the last 30 ms relative to the preceding 200 ms RMS.

    Computed at ``CLICK_ANALYSIS_RATE`` regardless of the input rate (see above).
    """
    window_sec = 0.25
    if len(audio) < int(sample_rate * window_sec):
        return 0.0
    window = audio[-int(sample_rate * window_sec) :]
    if sample_rate != CLICK_ANALYSIS_RATE:
        window = _resample_window(window, sample_rate, CLICK_ANALYSIS_RATE)
    rate = CLICK_ANALYSIS_RATE
    tail = window[-max(1, int(rate * 0.03)) :]
    context = window[-int(rate * 0.23) : -int(rate * 0.03)]
    if len(context) == 0:
        return 0.0
    tail_peak = float(np.max(np.abs(np.diff(tail)))) if len(tail) > 1 else 0.0
    context_rms = float(np.sqrt(np.mean(np.square(context)))) + 1e-8
    return tail_peak / context_rms


def _resample_window(window: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    """Resample a short excerpt without edge transients.

    ``resample_poly`` zero-pads its input, which would fabricate a step at the
    very end of the clip, exactly where the click detector looks. Reflect-pad
    both ends first and trim the padding afterwards.
    """
    pad = min(len(window) // 2, int(sample_rate * 0.02))
    if pad < 1:
        return resample_mono(window, sample_rate, target_rate)
    padded = np.pad(window, (pad, pad), mode="reflect")
    resampled = resample_mono(padded, sample_rate, target_rate)
    trim = int(round(pad * target_rate / sample_rate))
    return resampled[trim : len(resampled) - trim]
