"""Audio heuristics must not flip verdicts on padding, sample rate, or duration (issue #3)."""

import numpy as np
import pytest

from tts_assess.audio.features import (
    CLICK_ANALYSIS_RATE,
    TAIL_CLICK_THRESHOLD,
    _tail_click_score,
    analyze_features,
)
from tts_assess.config import OptionalMetricsConfig
from tts_assess.metrics import optional
from tts_assess.metrics.optional import SAMPLE_RATE, _vowel_prolongation, compute_optional_metrics


def _speech_like(sample_rate: int, seconds: float = 1.0, seed: int = 0) -> np.ndarray:
    """Amplitude-modulated noise standing in for speech (loud, broadband)."""
    rng = np.random.default_rng(seed)
    n = int(sample_rate * seconds)
    # The envelope never drops below 20% so every frame stays above -20 dB re speech.
    envelope = 0.6 + 0.4 * np.sin(np.linspace(0, 20 * np.pi, n))
    return (rng.standard_normal(n) * 0.1 * envelope).astype(np.float32)


# --- H8: silence ----------------------------------------------------------------


@pytest.mark.parametrize("pad_seconds", [0.5, 1.5, 5.0])
def test_appending_zeros_never_reduces_detected_silence(pad_seconds):
    sr = 24000
    rng = np.random.default_rng(1)
    hiss = rng.uniform(-0.001, 0.001, 3 * sr).astype(np.float32)
    clip = np.concatenate([_speech_like(sr), hiss])
    before = analyze_features(clip, sr, 1)
    padded = np.concatenate([clip, np.zeros(int(pad_seconds * sr), np.float32)])
    after = analyze_features(padded, sr, 1)
    assert after.silence_ratio >= before.silence_ratio
    assert after.trailing_silence_sec >= before.trailing_silence_sec + pad_seconds - 1e-3
    # The reporter's reversal: a failing silence ratio must stay failing.
    assert before.silence_ratio > 0.65 and after.silence_ratio > 0.65


def test_quiet_hiss_after_speech_counts_as_silence():
    sr = 24000
    rng = np.random.default_rng(2)
    hiss = rng.uniform(-0.001, 0.001, 2 * sr).astype(np.float32)
    clip = np.concatenate([_speech_like(sr, 2.0), hiss])
    features = analyze_features(clip, sr, 1)
    assert 0.45 < features.silence_ratio < 0.55
    assert 1.9 < features.trailing_silence_sec <= 2.0


def test_all_zero_clip_is_fully_silent():
    features = analyze_features(np.zeros(24000, np.float32), 24000, 1)
    assert features.silence_ratio == 1.0
    assert features.trailing_silence_sec == 1.0


# --- H9: tail click --------------------------------------------------------------


def _pulse_clip(sample_rate: int, *, pulse_peak: float = 0.02) -> np.ndarray:
    """300 ms quiet tone with a smooth Gaussian pulse 10 ms before the end."""
    t = np.arange(int(0.3 * sample_rate)) / sample_rate
    background = np.sqrt(2) * 0.0005 * np.sin(2 * np.pi * 300 * t)
    pulse = pulse_peak * np.exp(-0.5 * ((t - (0.3 - 0.010)) / 0.0002) ** 2)
    return (background + pulse).astype(np.float32)


def test_tail_click_score_is_sample_rate_invariant():
    scores = {sr: _tail_click_score(_pulse_clip(sr), sr) for sr in (16000, 24000, 44100, 48000)}
    reference = scores[CLICK_ANALYSIS_RATE]
    for sr, score in scores.items():
        assert score == pytest.approx(reference, rel=0.15), (sr, scores)
    # And the verdict agrees across rates.
    verdicts = {score >= TAIL_CLICK_THRESHOLD for score in scores.values()}
    assert len(verdicts) == 1


def test_tail_click_still_detects_real_step_at_any_rate():
    for sr in (24000, 48000):
        t = np.arange(int(0.3 * sr)) / sr
        clip = (np.sqrt(2) * 0.0005 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        clip[-int(0.01 * sr) :] += 0.02  # a genuine DC step 10 ms before the end
        assert _tail_click_score(clip, sr) >= TAIL_CLICK_THRESHOLD


def test_clean_fade_out_has_no_click_at_any_rate():
    for sr in (24000, 44100, 48000):
        t = np.arange(int(0.5 * sr)) / sr
        tone = 0.1 * np.sin(2 * np.pi * 220 * t) * np.linspace(1.0, 0.0, len(t))
        assert _tail_click_score(tone.astype(np.float32), sr) < TAIL_CLICK_THRESHOLD


# --- H11: vowel prolongation -------------------------------------------------------


def _tone_clip(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    tone = (
        0.2 * np.sin(2 * np.pi * 200 * t)
        + 0.1 * np.sin(2 * np.pi * 600 * t)
        + 0.05 * np.sin(2 * np.pi * 1000 * t)
    )
    silence = np.zeros(SAMPLE_RATE, np.float32)
    return np.concatenate([silence, tone.astype(np.float32), silence])


def test_longer_sustained_vowel_never_scores_lower():
    scores = [_vowel_prolongation(_tone_clip(s))["vowel_prolongation_score"] for s in (1, 2, 4)]
    assert scores == sorted(scores)
    for seconds, score in zip((1, 2, 4), scores, strict=True):
        assert score == pytest.approx(seconds, abs=0.1)
        assert _vowel_prolongation(_tone_clip(seconds))["vowel_prolongation_detected"] is True


def test_short_vowel_is_not_flagged():
    result = _vowel_prolongation(_tone_clip(0.3))
    assert result["vowel_prolongation_score"] < 0.8
    assert result["vowel_prolongation_detected"] is False


# --- H7: NISQA input rate -----------------------------------------------------------


def test_nisqa_receives_native_rate_audio_and_heuristics_get_16k(monkeypatch):
    seen = {}

    def fake_nisqa(audio, sample_rate):
        seen["nisqa"] = (len(audio), sample_rate)
        return {"nisqa_mos": 4.0}

    def fake_vowel(audio):
        seen["vowel"] = len(audio)
        return {"vowel_prolongation_score": 0.0, "vowel_prolongation_detected": False}

    monkeypatch.setattr(optional, "_nisqa_v2", fake_nisqa)
    monkeypatch.setattr(optional, "_vowel_prolongation", fake_vowel)
    clip = _speech_like(48000, 1.0)
    options = OptionalMetricsConfig(nisqa_v2=True, vowel_prolongation=True)
    metrics = compute_optional_metrics(clip, None, options, sample_rate=48000)
    assert metrics["nisqa_mos"] == 4.0
    assert seen["nisqa"] == (48000, 48000)
    assert seen["vowel"] == 16000


def test_nisqa_model_instance_is_keyed_by_rate(monkeypatch):
    optional._nisqa_metric.cache_clear()
    created = []

    class FakeNisqa:
        def __init__(self, fs):
            created.append(fs)

    import sys
    import types

    torchmetrics = types.ModuleType("torchmetrics")
    audio_mod = types.ModuleType("torchmetrics.audio")
    nisqa_mod = types.ModuleType("torchmetrics.audio.nisqa")
    nisqa_mod.NonIntrusiveSpeechQualityAssessment = FakeNisqa
    monkeypatch.setitem(sys.modules, "torchmetrics", torchmetrics)
    monkeypatch.setitem(sys.modules, "torchmetrics.audio", audio_mod)
    monkeypatch.setitem(sys.modules, "torchmetrics.audio.nisqa", nisqa_mod)
    try:
        optional._nisqa_metric(24000)
        optional._nisqa_metric(48000)
        optional._nisqa_metric(24000)
    finally:
        optional._nisqa_metric.cache_clear()
    assert created == [24000, 48000]
