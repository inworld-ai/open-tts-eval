import wave
from pathlib import Path

from tts_assess.audio import read_audio
from tts_assess.config import AssessmentConfig
from tts_assess.io.manifest import ManifestItem
from tts_assess.pipeline import _cache_key, run_assessment


def _write_tone(
    path: Path,
    *,
    sample_rate: int = 16000,
    frame_count: int = 16000,
    amplitude: float = 0.1,
):
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        frame = int(amplitude * 32767).to_bytes(2, "little", signed=True)
        f.writeframes(frame * frame_count)


def _manifest(tmp_path: Path) -> Path:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    for name in ("a", "b"):
        _write_tone(audio_dir / f"{name}.wav")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        '{"id":"a","text":"Hello world.","audio_path":"audio/a.wav","speaker_id":"v1"}\n'
        '{"id":"b","text":"Second one.","audio_path":"audio/b.wav","speaker_id":"v1"}\n'
    )
    return manifest


def _config():
    config = AssessmentConfig.default()
    config.asr.backend = "mock"
    config.reporting.bootstrap_resamples = 100
    return config


def test_second_run_hits_cache(tmp_path: Path):
    manifest = _manifest(tmp_path)
    config = _config()

    _, first = run_assessment(manifest, tmp_path / "out", config)
    assert first["cache"]["hits"] == 0
    assert first["cache"]["misses"] == 2
    assert first["cache"]["dir"] == ".measure_cache"  # portable: relative to the output dir

    _, second = run_assessment(manifest, tmp_path / "out", config)
    assert second["cache"]["hits"] == 2
    assert second["cache"]["misses"] == 0
    # Cached rows produce identical measurements.
    assert second["metrics"]["wer"]["mean"] == first["metrics"]["wer"]["mean"]


def test_no_cache_disables_caching(tmp_path: Path):
    manifest = _manifest(tmp_path)
    _, summary = run_assessment(manifest, tmp_path / "out", _config(), use_cache=False)
    assert "cache" not in summary
    assert not (tmp_path / "out" / ".measure_cache").exists()


def test_audio_path_relative_when_under_output_dir(tmp_path: Path):
    import json

    # Audio colocated inside the output dir -> results.jsonl audio_path is relative.
    out = tmp_path / "run"
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True)
    _write_tone(audio_dir / "a.wav")
    manifest = out / "manifest.jsonl"
    manifest.write_text('{"id":"a","text":"Hi.","audio_path":"audio/a.wav","speaker_id":"v"}\n')

    run_assessment(manifest, out, _config())
    row = json.loads((out / "results.jsonl").read_text().splitlines()[0])
    assert row["audio_path"] == "audio/a.wav"


def test_audio_path_absolute_when_outside_output_dir(tmp_path: Path):
    import json

    manifest = _manifest(tmp_path)  # audio under tmp_path/audio, output elsewhere
    run_assessment(manifest, tmp_path / "out", _config())
    row = json.loads((tmp_path / "out" / "results.jsonl").read_text().splitlines()[0])
    assert row["audio_path"].endswith(".wav")
    assert Path(row["audio_path"]).is_absolute()


def test_config_change_invalidates_cache(tmp_path: Path):
    manifest = _manifest(tmp_path)
    run_assessment(manifest, tmp_path / "out", _config())
    # Turning on an optional metric changes the fingerprint -> recompute.
    changed = _config()
    changed.optional_metrics.voice_lens = True
    _, summary = run_assessment(manifest, tmp_path / "out", changed)
    assert summary["cache"]["misses"] == 2
    assert "expressiveness_proxy" in summary["metrics"]


def test_audio_sample_rate_and_channels_are_part_of_cache_key(tmp_path: Path):
    manifest = _manifest(tmp_path)
    run_assessment(manifest, tmp_path / "out", _config())

    # Preserve identical decoded samples while changing only the WAV sample-rate header.
    _write_tone(tmp_path / "audio" / "a.wav", sample_rate=8000)
    _, summary = run_assessment(manifest, tmp_path / "out", _config())
    assert summary["cache"]["hits"] == 1
    assert summary["cache"]["misses"] == 1


def test_reference_audio_content_is_part_of_cache_key(tmp_path: Path):
    audio_path = tmp_path / "audio.wav"
    reference_path = tmp_path / "reference.wav"
    _write_tone(audio_path)
    _write_tone(reference_path, amplitude=0.1)
    audio, sample_rate, channels = read_audio(audio_path)
    item = ManifestItem(
        id="a",
        text="Hello",
        audio_path=audio_path,
        reference_audio_path=reference_path,
    )
    config = _config()
    config.optional_metrics.speaker_similarity = True

    first = _cache_key(audio, sample_rate, channels, item, config)
    _write_tone(reference_path, amplitude=0.2)
    second = _cache_key(audio, sample_rate, channels, item, config)
    assert first != second
