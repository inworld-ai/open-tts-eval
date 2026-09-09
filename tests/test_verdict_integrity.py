"""Failure states must survive reporting, and coverage must be visible (issue #3, H1/H10)."""

import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tts_assess.audio import read_audio
from tts_assess.config import AssessmentConfig, BoundThreshold
from tts_assess.pipeline import run_assessment
from tts_assess.reporting.aggregate import summarize
from tts_assess.reporting.compare import build_comparison, build_run_report
from tts_assess.reporting.comparison_html import render_comparison_html
from tts_assess.reporting.thresholds import classify_row, evaluate_thresholds, measurement_failure
from tts_assess.reporting.writers import write_jsonl, write_summary


def _config():
    config = AssessmentConfig.default()
    config.reporting.bootstrap_resamples = 100
    return config


def _cohort(prefix: str, n: int = 10):
    """n clips over one voice with distinct texts (comparable across runs)."""
    return [
        {
            "id": f"{prefix}_{i}",
            "speaker_id": "v",
            "text": f"sentence number {i}",
            "wer": 0.0,
            "cer": 0.0,
            "asr_backend": "mock",
            "asr_model": "mock",
        }
        for i in range(n)
    ]


def _health_row(report, key):
    for row in report["health_table"]["rows"]:
        if row["key"] == key:
            return row
    raise KeyError(key)


def _metric_row(report, key):
    for group in report["metric_table"]["groups"]:
        for row in group["rows"]:
            if row["key"] == key:
                return row
    raise KeyError(key)


# --- classify_row -----------------------------------------------------------


def test_terminal_errors_stay_failed_when_thresholds_are_reapplied():
    thresholds = _config().thresholds
    for key, stage in (("error_asr", "asr"), ("error_audio", "audio")):
        row = {"id": "x", "text": "hi", key: "boom"}
        status, metric_statuses, labels = classify_row(row, thresholds)
        assert status == "fail"
        assert labels == [f"fail:{stage}"]
        assert metric_statuses == {}
        assert measurement_failure(row) == stage
        assert measurement_failure(row, "wer") == stage


def test_heuristic_flags_appear_in_metric_statuses():
    thresholds = _config().thresholds
    row = {"id": "x", "text": "hi", "wer": 0.0, "repeated_span": True, "tail_hallucination": False}
    status, metric_statuses, labels = classify_row(row, thresholds)
    assert status == "fail"
    assert metric_statuses["repeated_span"] == "fail"
    assert metric_statuses["tail_hallucination"] == "pass"
    assert "fail:repeated_span" in labels


def test_optional_metric_error_is_a_measurement_failure_for_its_metrics():
    row = {"id": "x", "text": "hi", "wer": 0.0, "nisqa_error": "cuda exploded"}
    assert measurement_failure(row) is None  # not terminal
    assert measurement_failure(row, "nisqa_mos") == "nisqa"
    assert measurement_failure(row, "wer") is None
    # Errors do not turn the row into a fail on their own; the report shows them.
    status, _, _ = classify_row(row, _config().thresholds)
    assert status == "pass"


def test_nan_metric_fails_threshold_instead_of_passing():
    thresholds = {"wer": BoundThreshold(warn=0.05, fail=0.10)}
    overall, statuses, labels = evaluate_thresholds({"wer": float("nan")}, thresholds)
    assert overall == "fail"
    assert statuses["wer"] == "fail"
    assert labels == ["invalid:wer"]
    overall, _, _ = evaluate_thresholds({"wer": float("inf")}, thresholds)
    assert overall == "fail"


# --- reports ------------------------------------------------------------------


def test_comparison_keeps_error_rows_failed_and_counts_them_against_health():
    config = _config()
    good = _cohort("good")
    for row in good:
        row["nisqa_mos"] = 4.0
    sparse = _cohort("sparse")
    sparse[0]["nisqa_mos"] = 5.0
    for row in sparse[1:]:
        row["nisqa_error"] = "model crashed"
    sparse[-1] = {**sparse[-1], "error_asr": "whisper died"}
    del sparse[-1]["nisqa_error"]

    report = build_comparison([("good", good), ("sparse", sparse)], config)

    # The ASR-failed row is still a fail after reclassification.
    assert sparse[-1]["status"] == "fail" and sparse[-1]["failure_labels"] == ["fail:asr"]

    mos_health = _health_row(report, "nisqa_mos")
    good_cell, sparse_cell = mos_health["cells"]
    assert good_cell["good_rate"] == 1.0 and good_cell["measured"] == 10
    # 1 measured pass out of 10 attempted (8 NISQA errors + 1 ASR error).
    assert sparse_cell["good_rate"] == pytest.approx(0.1)
    assert sparse_cell["measured"] == 1 and sparse_cell["failed"] == 9
    assert sparse_cell["status"] == "fail"

    mos_metric = _metric_row(report, "nisqa_mos")
    good_metric, sparse_metric = mos_metric["cells"]
    assert good_metric["n"] == 10 and good_metric["total"] == 10 and good_metric["failed"] == 0
    assert sparse_metric["n"] == 1 and sparse_metric["total"] == 10 and sparse_metric["failed"] == 9

    html = render_comparison_html(report)
    assert "n=1/10 · 9 failed" in html


def test_all_failed_metric_family_is_still_listed_with_zero_coverage():
    config = _config()
    a = _cohort("a")
    b = _cohort("b")
    for row in a:
        row["nisqa_mos"] = 4.0
    for row in b:
        row["nisqa_error"] = "not installed"
    report = build_comparison([("a", a), ("b", b)], config)
    cell = _metric_row(report, "nisqa_mos")["cells"][1]
    assert cell["na"] is True and cell["n"] == 0 and cell["failed"] == 10
    assert _health_row(report, "nisqa_mos")["cells"][1]["good_rate"] == 0.0


def test_comparison_rejects_runs_scored_by_different_asr():
    a = _cohort("a")
    b = _cohort("b")
    for row in b:
        row["asr_backend"] = "faster-whisper"
        row["asr_model"] = "base.en"
    with pytest.raises(ValueError, match="asr_backend differs"):
        build_comparison([("a", a), ("b", b)], _config())


def test_comparison_rejects_runs_scored_by_different_evaluator_code():
    a = _cohort("a")
    b = _cohort("b")
    for row in a:
        row["evaluator"] = {"code_hash": "aaa", "normalization": "english-basic", "version": "1"}
    for row in b:
        row["evaluator"] = {"code_hash": "bbb", "normalization": "english-basic", "version": "1"}
    with pytest.raises(ValueError, match="code_hash differs"):
        build_comparison([("a", a), ("b", b)], _config())


def test_run_report_lists_measurement_failures_and_heuristics(tmp_path: Path):
    config = _config()
    rows = _cohort("r", 6)
    rows[0]["repeated_span"] = True
    rows[1] = {**rows[1], "error_asr": "whisper timeout"}
    rows[2]["nisqa_error"] = "oom"
    for row in rows:
        status, ms, labels = classify_row(row, config.thresholds)
        row["status"], row["metric_statuses"], row["failure_labels"] = status, ms, labels
    summary = summarize(rows, bootstrap_resamples=50, group_by_voice=False)
    assert summary["measurement_failures"] == {"asr": 1, "nisqa": 1}

    report = build_run_report(rows, summary, config, output_dir=tmp_path, title="T", label="r")
    violations = report["violations"]
    names = {c["metric"] for c in violations["counts"]}
    assert {"repeated span", "ASR failure", "NISQA failure"} <= names
    assert "1 violations across 1 metric(s)." in violations["state"]
    assert "2 measurement failure(s) across 2 stage(s)." in violations["state"]
    example_metrics = [e["metric"] for e in violations["examples"]["rows"]]
    assert "ASR failure" in example_metrics
    asr_example = next(e for e in violations["examples"]["rows"] if e["metric"] == "ASR failure")
    assert asr_example["value"] == "not measured" and "whisper timeout" in asr_example["heard"]
    assert set(asr_example) == {"metric", "voice", "value", "expected", "heard", "audio"}
    assert report["subtitle"] == "6 samples (1 not measured)"


# --- pipeline -------------------------------------------------------------------


def _write_tone(path: Path, sample_rate: int = 16000):
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        frame = int(0.1 * 32767).to_bytes(2, "little", signed=True)
        f.writeframes(frame * sample_rate)


def _manifest(tmp_path: Path) -> Path:
    (tmp_path / "audio").mkdir()
    _write_tone(tmp_path / "audio" / "a.wav")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text('{"id":"a","text":"Hello world.","audio_path":"audio/a.wav"}\n')
    return manifest


def test_failed_optional_metric_is_not_cached(tmp_path: Path, monkeypatch):
    manifest = _manifest(tmp_path)
    config = _config()
    config.asr.backend = "mock"
    config.optional_metrics.nisqa_v2 = True
    calls = []

    def flaky(audio, reference, options, **_kwargs):
        calls.append(1)
        return {"nisqa_error": "boom"} if len(calls) == 1 else {"nisqa_mos": 1.0}

    monkeypatch.setattr("tts_assess.evaluate.compute_optional_metrics", flaky)
    rows, first = run_assessment(manifest, tmp_path / "out", config)
    assert rows[0]["nisqa_error"] == "boom" and first["cache"]["misses"] == 1
    assert first["measurement_failures"] == {"nisqa": 1}

    rows, second = run_assessment(manifest, tmp_path / "out", config)
    # The repaired metric ran again instead of replaying the cached failure...
    assert len(calls) == 2 and second["cache"]["hits"] == 0
    assert rows[0]["nisqa_mos"] == 1.0 and rows[0]["status"] == "fail"

    # ...and the successful measurement is now cached.
    run_assessment(manifest, tmp_path / "out", config)
    assert len(calls) == 2


def test_nan_audio_is_rejected_at_decode(tmp_path: Path):
    path = tmp_path / "nan.wav"
    data = np.zeros(16000, dtype=np.float32)
    data[5000] = np.nan
    sf.write(path, data, 16000, subtype="FLOAT")
    with pytest.raises(ValueError, match="non-finite"):
        read_audio(path)

    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({"id": "n", "text": "x", "audio_path": "nan.wav"}) + "\n")
    config = _config()
    config.asr.backend = "mock"
    rows, summary = run_assessment(manifest, tmp_path / "out", config)
    assert rows[0]["status"] == "fail" and "non-finite" in rows[0]["error_audio"]
    assert summary["failed"] == 1
    # Nothing NaN reached the canonical record (a NaN literal is not valid JSON).
    for line in (tmp_path / "out" / "results.jsonl").read_text().splitlines():
        json.loads(line, parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))


def test_writers_refuse_nan(tmp_path: Path):
    with pytest.raises(ValueError):
        write_jsonl(tmp_path / "r.jsonl", [{"wer": math.nan}])
    with pytest.raises(ValueError):
        write_summary(tmp_path / "s.json", {"mean": math.inf})
