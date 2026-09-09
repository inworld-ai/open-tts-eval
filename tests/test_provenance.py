"""Evaluator provenance: stamped on rows, part of the cache key, checked when comparing."""

import importlib
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from tts_assess.config import AssessmentConfig
from tts_assess.io.manifest import ManifestItem
from tts_assess.pipeline import _cache_key, run_assessment
from tts_assess.provenance import (
    MEASUREMENT_MODULES,
    describe_normalization,
    evaluator_fingerprint,
    measurement_code_hash,
    plugin_hash,
)
from tts_assess.reporting.compare import build_comparison


def _config():
    config = AssessmentConfig.default()
    config.asr.backend = "mock"
    config.reporting.bootstrap_resamples = 50
    return config


def test_fingerprint_shape_and_determinism():
    fp = evaluator_fingerprint(_config())
    assert set(fp) == {"version", "code_hash", "plugin_hash", "asr", "normalization"}
    assert len(fp["code_hash"]) == 16 and fp["plugin_hash"] is None
    assert fp["asr"] == "mock:small" and fp["normalization"] == "english-basic"
    assert evaluator_fingerprint(_config()) == fp


def test_measurement_modules_exist_and_hash_covers_them():
    root = Path(__import__("tts_assess").__file__).parent
    for relative in MEASUREMENT_MODULES:
        assert (root / relative).exists(), relative
    assert measurement_code_hash() == measurement_code_hash()


def test_describe_normalization_variants():
    config = _config()
    config.normalization.expand_numbers = False
    assert describe_normalization(config) == "english-basic(no-number-expansion)"
    config.normalization.backend = "plugin"
    config.normalization.plugin = "mymod:norm"
    assert describe_normalization(config) == "plugin:mymod:norm"


def test_plugin_source_change_changes_hash_and_cache_key(tmp_path: Path, monkeypatch):
    plugin = tmp_path / "normplug_test.py"
    plugin.write_text("def norm(text):\n    return text.lower()\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("normplug_test", None)
    config = _config()
    config.normalization.backend = "plugin"
    config.normalization.plugin = "normplug_test:norm"

    audio = np.zeros(1600, dtype=np.float32)
    item = ManifestItem(id="a", text="Hello", audio_path=tmp_path / "a.wav")
    first_hash = plugin_hash(config)
    first_key = _cache_key(audio, 16000, 1, item, config)

    # Same plugin name, different implementation (a helper changed, say).
    plugin.write_text("def norm(text):\n    return text.upper()\n")
    importlib.invalidate_caches()
    importlib.reload(sys.modules["normplug_test"])
    second_hash = plugin_hash(config)
    second_key = _cache_key(audio, 16000, 1, item, config)

    assert first_hash and second_hash and first_hash != second_hash
    assert first_key != second_key
    sys.modules.pop("normplug_test", None)


def test_cache_key_depends_on_measurement_code_hash(tmp_path: Path):
    audio = np.zeros(1600, dtype=np.float32)
    item = ManifestItem(id="a", text="Hello", audio_path=tmp_path / "a.wav")
    base = {"plugin_hash": None}
    key_a = _cache_key(audio, 16000, 1, item, _config(), {**base, "code_hash": "aaaa"})
    key_b = _cache_key(audio, 16000, 1, item, _config(), {**base, "code_hash": "bbbb"})
    assert key_a != key_b


def _write_tone(path: Path):
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(int(0.1 * 32767).to_bytes(2, "little", signed=True) * 16000)


def test_run_stamps_rows_and_summary_with_evaluator(tmp_path: Path):
    (tmp_path / "audio").mkdir()
    _write_tone(tmp_path / "audio" / "a.wav")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text('{"id":"a","text":"Hello world.","audio_path":"audio/a.wav"}\n')
    rows, summary = run_assessment(manifest, tmp_path / "out", _config())
    assert rows[0]["evaluator"] == summary["evaluator"]
    assert rows[0]["evaluator"]["code_hash"] == measurement_code_hash()
    assert rows[0]["asr_backend"] == "mock"


def test_comparison_accepts_same_evaluator_and_rejects_plugin_drift():
    def rows(prefix, plugin):
        return [
            {
                "id": f"{prefix}_{i}",
                "speaker_id": "v",
                "text": f"text {i}",
                "wer": 0.0,
                "asr_backend": "mock",
                "asr_model": "mock",
                "evaluator": {"code_hash": "c", "plugin_hash": plugin, "normalization": "n"},
            }
            for i in range(4)
        ]

    config = _config()
    build_comparison([("a", rows("a", None)), ("b", rows("b", None))], config)
    with pytest.raises(ValueError, match="plugin_hash differs"):
        build_comparison([("a", rows("a", "p1")), ("b", rows("b", "p2"))], config)
