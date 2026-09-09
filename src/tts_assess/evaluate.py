"""Per-pair evaluation core shared by the CLI pipeline and in-memory callers.

``measure_pair`` computes the cacheable measurement for one audio+text pair;
``evaluate_pair`` adds threshold classification and the evaluator stamp. Both
operate on in-memory audio (container bytes or a decoded array), so a service
can call them without touching disk, and the CLI pipeline delegates here:
parity by construction rather than by keeping two code paths in step.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from tts_assess.asr import transcribe
from tts_assess.audio import analyze_features, read_audio_bytes
from tts_assess.config import AssessmentConfig
from tts_assess.metrics import audio_metric_dict, compute_text_metrics
from tts_assess.metrics.optional import compute_optional_metrics
from tts_assess.normalization import build_normalizer
from tts_assess.provenance import evaluator_fingerprint
from tts_assess.reporting.thresholds import classify_row


def measure_pair(
    audio: bytes | np.ndarray,
    text: str,
    config: AssessmentConfig,
    *,
    sample_rate: int | None = None,
    channels: int = 1,
    reference_audio: bytes | np.ndarray | None = None,
    reference_sample_rate: int | None = None,
    normalizer: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Measure one pair: audio features, ASR, text metrics, optional metrics.

    ``audio`` is either encoded container bytes (WAV/FLAC/...) or a decoded mono
    float32 array (then ``sample_rate`` is required). Failures are reported the
    way the pipeline records them: a dict with ``error_audio``/``error_asr``,
    ``status: "fail"`` and a failure label, never an exception for bad input.
    """
    if isinstance(audio, bytes | bytearray):
        try:
            audio, sample_rate, channels = read_audio_bytes(bytes(audio))
        except Exception as exc:
            return _terminal_error("audio", exc)
    elif sample_rate is None:
        raise ValueError("sample_rate is required when audio is a decoded array")
    if normalizer is None:
        normalizer = build_normalizer(config.normalization)

    measured: dict[str, Any] = {}
    try:
        audio_features = analyze_features(audio, sample_rate, channels)
        measured.update(audio_metric_dict(audio_features, text))
    except Exception as exc:
        return _terminal_error("audio", exc)

    # The decoded array goes to ASR (resampled inside `transcribe`), so file and
    # in-memory callers transcribe byte-identical input.
    try:
        transcript = transcribe(audio, config.asr, expected_text=text, sample_rate=sample_rate)
        measured.update(
            {
                "transcript": transcript.text,
                "asr_backend": transcript.backend,
                "asr_model": transcript.model,
                "asr_segments": [segment.__dict__ for segment in transcript.segments],
            }
        )
    except Exception as exc:
        return _terminal_error("asr", exc)

    normalized_text = normalizer(text)
    normalized_transcript = normalizer(measured["transcript"])
    measured["normalized_text"] = normalized_text
    measured["normalized_transcript"] = normalized_transcript
    measured.update(compute_text_metrics(normalized_text, normalized_transcript).to_dict())

    reference: np.ndarray | None = None
    if reference_audio is not None:
        try:
            if isinstance(reference_audio, bytes | bytearray):
                reference, reference_sample_rate, _ = read_audio_bytes(bytes(reference_audio))
            elif reference_sample_rate is not None:
                reference = reference_audio
        except Exception:
            reference = None
    measured.update(
        compute_optional_metrics(
            audio,
            reference,
            config.optional_metrics,
            sample_rate=sample_rate,
            reference_sample_rate=reference_sample_rate,
        )
    )
    return measured


def evaluate_pair(
    audio: bytes | np.ndarray,
    text: str,
    config: AssessmentConfig | None = None,
    *,
    sample_rate: int | None = None,
    channels: int = 1,
    reference_audio: bytes | np.ndarray | None = None,
    reference_sample_rate: int | None = None,
    normalizer: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Evaluate one audio+text pair: measurement, thresholds, evaluator stamp.

    The public entry point for callers outside the manifest pipeline. Returns
    a row shaped like a ``results.jsonl`` line (minus id/paths): the measured
    metrics with ``status``, ``metric_statuses`` and ``failure_labels`` applied
    from ``config.thresholds``, and the ``evaluator`` provenance block.
    """
    config = config or AssessmentConfig.default()
    row: dict[str, Any] = {"text": text, "evaluator": evaluator_fingerprint(config)}
    row.update(
        measure_pair(
            audio,
            text,
            config,
            sample_rate=sample_rate,
            channels=channels,
            reference_audio=reference_audio,
            reference_sample_rate=reference_sample_rate,
            normalizer=normalizer,
        )
    )
    status, metric_statuses, labels = classify_row(row, config.thresholds)
    row["status"] = status
    row["metric_statuses"] = metric_statuses
    row["failure_labels"] = labels
    return row


def _terminal_error(stage: str, exc: Exception) -> dict[str, Any]:
    return {f"error_{stage}": str(exc), "status": "fail", "failure_labels": [f"fail:{stage}"]}
