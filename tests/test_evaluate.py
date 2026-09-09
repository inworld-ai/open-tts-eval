"""The in-memory per-pair core must match the manifest pipeline row for row."""

import io
import wave
from pathlib import Path

import numpy as np
import pytest

from tts_assess.asr import transcribe
from tts_assess.config import AssessmentConfig
from tts_assess.evaluate import evaluate_pair, measure_pair
from tts_assess.pipeline import run_assessment


def _mock_config() -> AssessmentConfig:
    config = AssessmentConfig.default()
    config.asr.backend = "mock"
    config.asr.model = "mock"
    config.optional_metrics.vowel_prolongation = True
    config.optional_metrics.voice_lens = True
    return config


def _tone_wav_bytes(sample_rate: int = 16000, seconds: float = 1.0) -> bytes:
    samples = [int(0.1 * 32767)] * int(sample_rate * seconds)
    buffer = io.BytesIO()
    with wave.open(buffer, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples))
    return buffer.getvalue()


def test_evaluate_pair_from_bytes_returns_metrics_status_and_provenance():
    row = evaluate_pair(_tone_wav_bytes(), "Hello world.", _mock_config())

    assert row["status"] in {"pass", "warn", "fail"}
    assert row["wer"] == 0
    assert row["transcript"] == "Hello world."
    assert row["duration_sec"] > 0.9
    assert "metric_statuses" in row and "failure_labels" in row
    assert row["evaluator"]["asr"] == "mock:mock"


def test_evaluate_pair_accepts_decoded_array():
    audio = np.full(16000, 0.1, dtype=np.float32)
    row = evaluate_pair(audio, "Hello world.", _mock_config(), sample_rate=16000)

    assert row["wer"] == 0
    assert row["sample_rate"] == 16000


def test_evaluate_pair_reports_undecodable_audio_as_failure():
    row = evaluate_pair(b"not audio at all", "Hello world.", _mock_config())

    assert row["status"] == "fail"
    assert "error_audio" in row
    assert row["failure_labels"] == ["fail:audio"]


def test_measure_pair_requires_sample_rate_for_arrays():
    with pytest.raises(ValueError, match="sample_rate"):
        measure_pair(np.zeros(16000, dtype=np.float32), "hi", _mock_config())


def test_transcribe_array_requires_sample_rate_for_whisper():
    config = AssessmentConfig.default().asr  # faster-whisper backend
    # Checked before any model is loaded, so this needs neither the dependency
    # nor the network.
    with pytest.raises(ValueError, match="sample_rate"):
        transcribe(np.zeros(16000, dtype=np.float32), config)


def test_whisper_decoder_is_seeded_per_clip(monkeypatch):
    import sys
    import types

    from tts_assess.asr import backends

    seeds: list[int] = []
    fake_ct2 = types.ModuleType("ctranslate2")
    fake_ct2.set_random_seed = seeds.append
    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ct2)

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            return iter(()), None

    monkeypatch.setattr(backends, "_load_whisper_model", lambda *a: FakeModel())
    config = AssessmentConfig.default().asr
    for _ in range(2):
        result = transcribe(np.zeros(16000, dtype=np.float32), config, sample_rate=16000)
        assert result.backend == "faster-whisper" and result.text == ""
    assert seeds == [backends.ASR_DECODER_SEED, backends.ASR_DECODER_SEED]


def test_evaluate_pair_matches_pipeline_row(tmp_path: Path):
    wav = _tone_wav_bytes()
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "sample.wav").write_bytes(wav)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"id":"sample","text":"Hello world.","audio_path":"audio/sample.wav"}\n')
    config = _mock_config()

    rows, _ = run_assessment(manifest, tmp_path / "out", config, use_cache=False)
    pipeline_row = rows[0]
    pair_row = evaluate_pair(wav, "Hello world.", config)

    for key in (
        "wer",
        "cer",
        "transcript",
        "normalized_transcript",
        "duration_sec",
        "silence_ratio",
        "clipping_ratio",
        "tail_click_score",
        "chars_per_second",
        "vowel_prolongation_score",
        "expressiveness_proxy",
        "status",
        "failure_labels",
        "evaluator",
    ):
        assert pair_row[key] == pipeline_row[key], key
    assert pair_row["metric_statuses"] == pipeline_row["metric_statuses"]
