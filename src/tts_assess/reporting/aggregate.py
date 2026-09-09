from __future__ import annotations

import math
from typing import Any

from tts_assess.reporting.stats import summarize_values
from tts_assess.reporting.thresholds import (
    METRIC_ERROR_KEYS,
    failed_stages,
    measurement_failure,
)

UNSPECIFIED_VOICE = "unspecified"

# Rows sharing a text are not independent (same content, different voices), so
# the bootstrap resamples texts as clusters. Falls back to i.i.d. automatically
# when every row has a distinct text.
CLUSTER_KEY = "text"


def summarize(
    rows: list[dict[str, Any]],
    *,
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 2000,
    group_by_voice: bool = True,
) -> dict[str, Any]:
    summary = _group_summary(rows, confidence_level, bootstrap_resamples)

    failures: dict[str, int] = {}
    for row in rows:
        for label in row.get("failure_labels", []):
            failures[label] = failures.get(label, 0) + 1
    summary["failure_counts"] = dict(
        sorted(failures.items(), key=lambda item: item[1], reverse=True)
    )

    if group_by_voice:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            voice = row.get("speaker_id") or UNSPECIFIED_VOICE
            groups.setdefault(voice, []).append(row)
        summary["by_voice"] = {
            voice: _group_summary(voice_rows, confidence_level, bootstrap_resamples)
            for voice, voice_rows in sorted(groups.items())
        }
    return summary


def _group_summary(
    rows: list[dict[str, Any]],
    confidence_level: float,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    total = len(rows)
    passed = sum(1 for row in rows if row.get("status") == "pass")
    warned = sum(1 for row in rows if row.get("status") == "warn")
    failed = sum(1 for row in rows if row.get("status") == "fail")

    metric_summary: dict[str, Any] = {}
    for metric in _numeric_metric_names(rows):
        measured = [row for row in rows if metric_value(row, metric) is not None]
        values = [metric_value(row, metric) for row in measured]
        clusters = [str(row.get(CLUSTER_KEY)) for row in measured]
        interval = summarize_values(
            values,
            confidence=confidence_level,
            resamples=bootstrap_resamples,
            clusters=clusters if any(clusters) else None,
        )
        metric_summary[metric] = {
            "count": interval["n"],
            # Rows that attempted this metric's stage and errored (or never got
            # that far). Shown next to the mean so a high score over few clips
            # cannot masquerade as a healthy run.
            "failed": sum(1 for row in rows if measurement_failure(row, metric) is not None),
            "mean": interval["mean"],
            "std": interval["std"],
            "median": interval["median"],
            "p95": interval["p95"],
            "min": interval["min"],
            "max": interval["max"],
            "ci_low": interval["ci_low"],
            "ci_high": interval["ci_high"],
            "ci_level": interval["ci_level"],
            "ci_method": interval["ci_method"],
        }

    return {
        "sample_count": total,
        "passed": passed,
        "warned": warned,
        "failed": failed,
        "pass_rate": passed / total if total else 0.0,
        "metrics": metric_summary,
        "violations": _violation_counts(rows),
        "measurement_failures": _measurement_failure_counts(rows),
    }


def metric_value(row: dict[str, Any], metric: str) -> float | None:
    """The row's finite numeric value for ``metric``, else None (bools excluded)."""
    value = row.get(metric)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if not math.isfinite(float(value)):
        return None
    return float(value)


def _measurement_failure_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Rows per failed measurement stage (audio, asr, nisqa, ...)."""
    counts: dict[str, int] = {}
    for row in rows:
        for stage in failed_stages(row):
            counts[stage] = counts.get(stage, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def _violation_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per-metric warn/fail counts derived from each row's threshold statuses."""
    violations: dict[str, dict[str, int]] = {}
    for row in rows:
        for metric, status in (row.get("metric_statuses") or {}).items():
            if status in ("warn", "fail"):
                bucket = violations.setdefault(metric, {"warn": 0, "fail": 0, "total": 0})
                bucket[status] += 1
                bucket["total"] += 1
    return dict(sorted(violations.items(), key=lambda item: item[1]["total"], reverse=True))


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires non-empty values")
    index = (len(ordered) - 1) * p / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _numeric_metric_names(rows: list[dict[str, Any]]) -> list[str]:
    excluded = {"sample_rate", "channels"}
    names: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key in excluded:
                continue
            if isinstance(value, int | float) and not isinstance(value, bool):
                names.add(key)
        # A stage that failed on every row leaves no numeric value behind; still
        # list its metrics so the report can show "0 measured, N failed".
        for error_key, (_stage, metrics) in METRIC_ERROR_KEYS.items():
            if row.get(error_key) is not None:
                names.update(metrics)
    return sorted(names)
