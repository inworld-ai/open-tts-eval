from __future__ import annotations

import math
from typing import Any

from tts_assess.config import BoundThreshold, Status

# Boolean row signals that count as hard failures regardless of numeric bands.
HEURISTIC_FAIL_KEYS = (
    "empty_transcript",
    "tail_hallucination",
    "repeated_span",
    "tail_click_detected",
)

# Terminal measurement errors: the row never reached scoring. Key -> stage name.
TERMINAL_ERROR_KEYS: dict[str, str] = {"error_audio": "audio", "error_asr": "asr"}

# Optional-metric errors -> (stage name, the metrics that stage would have produced).
# A row carrying one of these keys attempted the stage and failed, which is
# different from a row that never requested it.
METRIC_ERROR_KEYS: dict[str, tuple[str, tuple[str, ...]]] = {
    "nisqa_error": (
        "nisqa",
        (
            "nisqa_mos",
            "nisqa_noisiness",
            "nisqa_discontinuity",
            "nisqa_coloration",
            "nisqa_loudness",
        ),
    ),
    "speaker_similarity_error": ("speaker_similarity", ("speaker_similarity",)),
}

_METRIC_TO_ERROR_KEY: dict[str, str] = {
    metric: key for key, (_stage, metrics) in METRIC_ERROR_KEYS.items() for metric in metrics
}


def measurement_failure(row: dict[str, Any], metric: str | None = None) -> str | None:
    """Why ``metric`` (or the whole row) has no measurement, or None if it was measured.

    Returns the failed stage name: ``audio``/``asr`` for terminal errors (every
    metric is missing), or the optional stage (``nisqa``, ``speaker_similarity``)
    whose error key is present on the row while its metric is absent.
    """
    for key, stage in TERMINAL_ERROR_KEYS.items():
        if row.get(key) is not None:
            return stage
    if metric is None:
        return None
    error_key = _METRIC_TO_ERROR_KEY.get(metric)
    if error_key is not None and row.get(error_key) is not None and row.get(metric) is None:
        return METRIC_ERROR_KEYS[error_key][0]
    return None


def failed_stages(row: dict[str, Any]) -> list[str]:
    """Every measurement stage that failed on this row (terminal or optional)."""
    terminal = measurement_failure(row)
    if terminal is not None:
        return [terminal]
    return [stage for key, (stage, _m) in METRIC_ERROR_KEYS.items() if row.get(key) is not None]


def classify_row(
    row: dict[str, object],
    thresholds: dict[str, BoundThreshold],
) -> tuple[Status, dict[str, Status], list[str]]:
    """Overall status, per-metric statuses, and failure labels for one row.

    Shared by the assessment pipeline and the cross-run comparison so both apply
    thresholds identically. Rows whose measurement failed terminally stay failed:
    re-applying thresholds must never turn an unmeasured clip into a pass.
    """
    terminal = measurement_failure(row)
    if terminal is not None:
        return "fail", {}, [f"fail:{terminal}"]
    labels: list[str] = [f"fail:{key}" for key in HEURISTIC_FAIL_KEYS if row.get(key)]
    status, metric_statuses, threshold_labels = evaluate_thresholds(row, thresholds)
    # Heuristic flags are verdicts too: surface them per metric so the health
    # table and the violations chapter can count them, not only the row status.
    for key in HEURISTIC_FAIL_KEYS:
        if row.get(key):
            metric_statuses[key] = "fail"
    labels.extend(threshold_labels)
    if any(label.startswith(("fail:", "invalid:")) for label in labels):
        status = "fail"
    return status, metric_statuses, sorted(set(labels))


def evaluate_thresholds(
    metrics: dict[str, object],
    thresholds: dict[str, BoundThreshold],
) -> tuple[Status, dict[str, Status], list[str]]:
    statuses: dict[str, Status] = {}
    labels: list[str] = []
    for name, threshold in thresholds.items():
        if name not in metrics or metrics[name] is None:
            continue
        value = metrics[name]
        if _is_non_finite(value):
            # NaN compares false against every bound; it must not pass silently.
            statuses[name] = "fail"
            labels.append(f"invalid:{name}")
            continue
        status = _evaluate_metric(value, threshold)
        statuses[name] = status
        if status in {"warn", "fail"}:
            labels.append(f"{status}:{name}")
    overall: Status = "pass"
    if any(status == "fail" for status in statuses.values()):
        overall = "fail"
    elif any(status == "warn" for status in statuses.values()):
        overall = "warn"
    return overall, statuses, labels


def _is_non_finite(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and not math.isfinite(float(value))
    )


def _evaluate_metric(value: object, threshold: BoundThreshold) -> Status:
    if isinstance(value, bool):
        if threshold.fail_if_true and value:
            return "fail"
        return "pass"
    if not isinstance(value, int | float):
        return "skip"
    numeric = float(value)
    if threshold.fail is not None and numeric >= threshold.fail:
        return "fail"
    if threshold.fail_below is not None and numeric <= threshold.fail_below:
        return "fail"
    if threshold.warn is not None and numeric >= threshold.warn:
        return "warn"
    if threshold.warn_below is not None and numeric <= threshold.warn_below:
        return "warn"
    return "pass"
