from __future__ import annotations

import json
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tts_assess.config import AssessmentConfig
from tts_assess.reporting.aggregate import CLUSTER_KEY, metric_value, summarize
from tts_assess.reporting.comparison_html import render_comparison_html
from tts_assess.reporting.metrics_meta import KEY_METRICS, direction, display_name
from tts_assess.reporting.stats import intervals_separated, paired_difference_ci
from tts_assess.reporting.thresholds import classify_row, failed_stages, measurement_failure
from tts_assess.reporting.writers import write_summary

STATIC_TITLE = "Comparative Report"

# Metric Comparison table layout: (group title, colorize best/worst?, [(key, label)]).
# Only metrics present in the data are shown; anything not listed here is hidden.
COMPARISON_GROUPS: list[tuple[str, bool, list[tuple[str, str]]]] = [
    ("Accuracy", True, [
        ("wer", "WER"),
        ("insertion_rate", "WER.Insertions"),
        ("deletion_rate", "WER.Deletions"),
        ("substitution_rate", "WER.Substitutions"),
        ("cer", "CER"),
    ]),
    ("NISQAv2", True, [
        ("nisqa_mos", "MOS"),
        ("nisqa_coloration", "coloration"),
        ("nisqa_discontinuity", "discontinuity"),
        ("nisqa_loudness", "loudness"),
        ("nisqa_noisiness", "noisiness"),
    ]),
    ("Subjective", True, [
        ("chars_per_second", "Chars/sec"),
        ("arousal_proxy", "Arousal"),
        ("expressiveness_proxy", "Expressiveness"),
    ]),
    ("Silence", False, [
        ("silence_ratio", "Silence"),
        ("leading_silence_sec", "Lead silence (s)"),
        ("trailing_silence_sec", "Tail silence (s)"),
    ]),
]

# Metrics excluded from the Model Health (Table 2) grid.
HEALTH_HIDE: frozenset[str] = frozenset({"wer", "cer", "insertion_rate"})

# Human-readable names for failed measurement stages.
STAGE_NAMES: dict[str, str] = {
    "audio": "Audio decode failure",
    "asr": "ASR failure",
    "nisqa": "NISQA failure",
    "speaker_similarity": "Speaker-similarity failure",
}


def load_run_rows(path: Path) -> list[dict[str, Any]]:
    """Load one run's per-sample rows from its results.jsonl."""
    path = Path(path)
    jsonl = path if path.suffix == ".jsonl" else path / "results.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"no results.jsonl found for run: {path}")
    rows = []
    for line in jsonl.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def default_label(path: Path) -> str:
    path = Path(path)
    return path.stem if path.suffix == ".jsonl" else path.name


def run_comparison(
    run_paths: list[Path],
    output_dir: Path,
    config: AssessmentConfig,
    *,
    labels: list[str] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    if labels and len(labels) != len(run_paths):
        raise ValueError(
            f"got {len(labels)} labels for {len(run_paths)} runs; provide one --label per run"
        )
    runs = []
    for index, path in enumerate(run_paths):
        rows = load_run_rows(path)
        label = labels[index] if labels else default_label(path)
        runs.append((label, rows))

    report_data = build_comparison(runs, config, title=title)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(output_dir / "comparison.json", report_data)
    (output_dir / "comparison.html").write_text(
        render_comparison_html(report_data), encoding="utf-8"
    )
    return report_data


def build_run_report(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    config: AssessmentConfig,
    *,
    output_dir: Path,
    title: str,
    label: str,
) -> dict[str, Any]:
    """Single-run report: the comparison layout (one column) + a violations chapter.

    Rows are expected already classified by the pipeline under this config.
    """
    return {
        "title": title,
        "subtitle": _subtitle(rows),
        "question": config.reporting.question,
        "labels": [label],
        "metric_table": _metric_table([summary], config.reporting.confidence_level),
        "health_table": _health_table([rows], config),
        "violations": _run_violations(rows, config, Path(output_dir)),
    }


def _subtitle(rows: list[dict[str, Any]]) -> str:
    failed = sum(1 for row in rows if measurement_failure(row) is not None)
    if failed:
        return f"{len(rows)} samples ({failed} not measured)"
    return f"{len(rows)} samples"


def _run_violations(
    rows: list[dict[str, Any]], config: AssessmentConfig, output_dir: Path
) -> dict[str, Any]:
    per_metric_limit = config.reporting.max_violation_examples
    total_limit = config.reporting.max_worst_samples
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        for metric, status in (row.get("metric_statuses") or {}).items():
            if status in ("warn", "fail"):
                bucket = counts.setdefault(metric, {"warn": 0, "fail": 0, "total": 0})
                bucket[status] += 1
                bucket["total"] += 1
    counts = dict(sorted(counts.items(), key=lambda kv: kv[1]["total"], reverse=True))
    count_rows = [{"metric": display_name(m), **b} for m, b in counts.items()]

    # Measurement failures are not threshold violations, but a report that hides
    # them ("no threshold was violated") misleads. Count and exemplify them too.
    failures: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for stage in failed_stages(row):
            failures.setdefault(stage, []).append(row)
    failures = dict(sorted(failures.items(), key=lambda kv: len(kv[1]), reverse=True))
    failure_rows = [
        {"metric": STAGE_NAMES.get(stage, f"{stage} failure"), "warn": 0,
         "fail": len(stage_rows), "total": len(stage_rows)}
        for stage, stage_rows in failures.items()
    ]

    used: set[str] = set()  # sample ids already shown, so each appears once
    examples: list[dict[str, Any]] = []
    has_audio = False

    def add_example(row: dict[str, Any], metric_label: str, value: str, heard: str) -> None:
        nonlocal has_audio
        used.add(row.get("id"))
        audio = (
            _existing_audio(row.get("audio_path"), output_dir)
            if config.reporting.embed_audio
            else None
        )
        has_audio = has_audio or bool(audio)
        examples.append(
            {
                "metric": metric_label,
                "voice": row.get("speaker_id") or "—",
                "value": value,
                "expected": _snippet(row.get("normalized_text") or row.get("text") or ""),
                "heard": heard,
                "audio": audio,
            }
        )

    for metric in counts:
        if len(examples) >= total_limit:
            break
        offenders = [
            row
            for row in rows
            if (row.get("metric_statuses") or {}).get(metric) in ("warn", "fail")
            and row.get("id") not in used
        ]
        numeric = [row for row in offenders if metric_value(row, metric) is not None]
        if numeric:
            numeric.sort(key=lambda row: row.get(metric), reverse=direction(metric) != "higher")
            chosen = numeric[:per_metric_limit]
        else:
            chosen = offenders[:per_metric_limit]
        chosen = chosen[: total_limit - len(examples)]
        for row in chosen:
            heard = _snippet(row.get("normalized_transcript") or row.get("transcript") or "")
            add_example(row, display_name(metric), _fmt_value(row.get(metric)), heard)

    for stage, stage_rows in failures.items():
        if len(examples) >= total_limit:
            break
        chosen = [row for row in stage_rows if row.get("id") not in used][:per_metric_limit]
        chosen = chosen[: total_limit - len(examples)]
        for row in chosen:
            add_example(
                row,
                STAGE_NAMES.get(stage, f"{stage} failure"),
                "not measured",
                _snippet(_stage_error(row, stage)),
            )

    total = sum(b["total"] for b in counts.values())
    failed_total = sum(len(v) for v in failures.values())
    parts = []
    if count_rows:
        parts.append(f"{total} violations across {len(counts)} metric(s).")
    else:
        parts.append("No configured threshold was violated.")
    if failed_total:
        parts.append(f"{failed_total} measurement failure(s) across {len(failures)} stage(s).")
    return {
        "state": " ".join(parts),
        "counts": count_rows + failure_rows,
        "examples": {"has_audio": has_audio, "rows": examples},
    }


def _stage_error(row: dict[str, Any], stage: str) -> str:
    for key in (f"error_{stage}", f"{stage}_error"):
        if row.get(key) is not None:
            return str(row[key])
    return ""


def _existing_audio(audio_path: str | None, output_dir: Path) -> str | None:
    if not audio_path:
        return None
    path = Path(audio_path)
    full = path if path.is_absolute() else output_dir / path
    return audio_path if full.exists() else None


def _snippet(text: str, limit: int = 90) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _fmt_value(value: Any) -> str:
    if isinstance(value, bool):
        return "flagged" if value else "ok"
    if isinstance(value, int | float):
        return _num(value)
    return "" if value is None else str(value)


def build_comparison(
    runs: list[tuple[str, list[dict[str, Any]]]],
    config: AssessmentConfig,
    *,
    title: str | None = None,
    subtitle: str | None = None,
) -> dict[str, Any]:
    _validate_comparable_runs(runs)
    confidence = config.reporting.confidence_level
    resamples = config.reporting.bootstrap_resamples
    labels = [label for label, _ in runs]

    all_rows: list[list[dict[str, Any]]] = []
    summaries: list[dict[str, Any]] = []
    for _label, rows in runs:
        # Re-apply the shared thresholds so both tables are comparable even if the
        # runs were originally assessed with different bands.
        for row in rows:
            status, metric_statuses, failure_labels = classify_row(row, config.thresholds)
            row["status"] = status
            row["metric_statuses"] = metric_statuses
            row["failure_labels"] = failure_labels
        all_rows.append(rows)
        summaries.append(
            summarize(
                rows,
                confidence_level=confidence,
                bootstrap_resamples=resamples,
                group_by_voice=False,
            )
        )

    return {
        "title": title or STATIC_TITLE,
        "subtitle": subtitle,
        "question": config.reporting.question,
        "generated": datetime.now(timezone.utc).isoformat(),
        "labels": labels,
        "metric_table": _metric_table(
            summaries, confidence, runs_rows=all_rows, resamples=resamples
        ),
        "health_table": _health_table(all_rows, config),
    }


def _validate_comparable_runs(runs: list[tuple[str, list[dict[str, Any]]]]) -> None:
    """Require equivalent content, voice-sampling shapes and evaluators across runs."""
    if len(runs) < 2:
        raise ValueError("comparison requires at least two runs")

    reference_label, reference_rows = runs[0]
    reference_content, reference_speakers = _comparison_profile(reference_label, reference_rows)
    reference_evaluator = _evaluator_profile(reference_rows)
    for label, rows in runs[1:]:
        content, speakers = _comparison_profile(label, rows)
        if content != reference_content:
            missing = sum((reference_content - content).values())
            extra = sum((content - reference_content).values())
            raise ValueError(
                f"run {label!r} is not comparable to {reference_label!r}: "
                f"text/language cohort differs ({missing} missing, {extra} extra samples)"
            )
        if speakers != reference_speakers:
            raise ValueError(
                f"run {label!r} is not comparable to {reference_label!r}: "
                "per-speaker sampling profile differs"
            )
        _check_same_evaluator(reference_label, reference_evaluator, label, _evaluator_profile(rows))


# Row fields that identify the evaluator which produced the scores. Runs scored
# by different ASR models or normalizers measure different things, and the gap
# would be misread as a TTS difference.
_EVALUATOR_FIELDS = ("asr_backend", "asr_model")
# Stamped by the pipeline (see provenance.py). The package version is not
# checked: two versions with identical measurement code score identically.
_PROVENANCE_FIELDS = ("code_hash", "plugin_hash", "normalization")


def _evaluator_profile(rows: list[dict[str, Any]]) -> dict[str, set[Any]]:
    profile: dict[str, set[Any]] = {}
    for row in rows:
        evaluator = row.get("evaluator")
        if isinstance(evaluator, dict):
            for field in _PROVENANCE_FIELDS:
                if field in evaluator:
                    value = evaluator[field]
                    profile.setdefault(f"evaluator.{field}", set()).add(
                        "none" if value is None else value
                    )
        if measurement_failure(row) is not None:
            continue  # failed rows carry no ASR fields
        for field in _EVALUATOR_FIELDS:
            profile.setdefault(field, set()).add(row.get(field))
    return profile


def _check_same_evaluator(
    reference_label: str,
    reference: dict[str, set[Any]],
    label: str,
    candidate: dict[str, set[Any]],
) -> None:
    unverifiable: list[str] = []
    for field in sorted(set(reference) | set(candidate)):
        ref_values = reference.get(field, set()) - {None}
        cand_values = candidate.get(field, set()) - {None}
        if not ref_values or not cand_values:
            if field.startswith("evaluator."):
                unverifiable.append(field.removeprefix("evaluator."))
            continue
        if ref_values != cand_values:
            raise ValueError(
                f"run {label!r} is not comparable to {reference_label!r}: {field} differs "
                f"({sorted(map(str, ref_values))} vs {sorted(map(str, cand_values))}); "
                "re-evaluate both runs with the same evaluator before comparing"
            )
    if unverifiable:
        warnings.warn(
            f"runs {reference_label!r} and {label!r}: evaluator {', '.join(unverifiable)} "
            "not recorded on at least one run (scored before provenance stamping); "
            "cannot verify both were scored by the same evaluator",
            stacklevel=3,
        )


def _comparison_profile(
    label: str, rows: list[dict[str, Any]]
) -> tuple[Counter[tuple[str, str]], Counter[tuple[tuple[tuple[str, str], int], ...]]]:
    if not rows:
        raise ValueError(f"run {label!r} has no samples")

    content: Counter[tuple[str, str]] = Counter()
    by_speaker: dict[str, Counter[tuple[str, str]]] = {}
    seen_ids: set[str] = set()
    speaker_presence: list[bool] = []
    for index, row in enumerate(rows):
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"run {label!r} sample {index} has no non-empty id")
        if sample_id in seen_ids:
            raise ValueError(f"run {label!r} has duplicate sample id {sample_id!r}")
        seen_ids.add(sample_id)

        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"run {label!r} sample {sample_id!r} has no non-empty text")
        key = (text, str(row.get("language") or ""))
        content[key] += 1

        speaker = row.get("speaker_id")
        has_speaker = isinstance(speaker, str) and bool(speaker.strip())
        speaker_presence.append(has_speaker)
        if has_speaker:
            by_speaker.setdefault(speaker, Counter())[key] += 1

    if any(speaker_presence) and not all(speaker_presence):
        raise ValueError(f"run {label!r} mixes samples with and without speaker_id")

    speaker_profiles: Counter[tuple[tuple[tuple[str, str], int], ...]] = Counter()
    if all(speaker_presence):
        for samples in by_speaker.values():
            speaker_profiles[tuple(sorted(samples.items()))] += 1
    return content, speaker_profiles


def _metric_table(
    summaries: list[dict[str, Any]],
    confidence: float,
    *,
    runs_rows: list[list[dict[str, Any]]] | None = None,
    resamples: int = 2000,
) -> dict[str, Any]:
    present = _union_metrics(summaries)
    groups = []
    for title, colorize, members in COMPARISON_GROUPS:
        rows = []
        for key, label in members:
            if key not in present:
                continue
            stats = [summary.get("metrics", {}).get(key) for summary in summaries]
            totals = [summary.get("sample_count", 0) for summary in summaries]
            rows.append(
                {
                    "metric": label,
                    "key": key,
                    "better": direction(key) if colorize else None,
                    "cells": _metric_cells(
                        key, stats, totals, colorize=colorize, runs_rows=runs_rows,
                        confidence=confidence, resamples=resamples,
                    ),
                }
            )
        if rows:
            groups.append({"title": title, "colorize": colorize, "rows": rows})
    methods = {
        stat.get("ci_method")
        for summary in summaries
        for stat in summary.get("metrics", {}).values()
        if stat.get("ci_method")
    }
    return {
        "ci_label": f"{round(confidence * 100)}% CI",
        "ci_method": "cluster" if "cluster" in methods else "iid",
        "groups": groups,
    }


def _metric_cells(
    metric: str,
    stats: list[dict[str, Any] | None],
    totals: list[int],
    *,
    colorize: bool,
    runs_rows: list[list[dict[str, Any]]] | None,
    confidence: float,
    resamples: int,
) -> list[dict[str, Any]]:
    best, worst = _best_worst_stats(metric, stats) if colorize else (None, None)
    cells = []
    for index, stat in enumerate(stats):
        coverage = {
            "n": (stat or {}).get("count", 0),
            "failed": (stat or {}).get("failed", 0),
            "total": totals[index] if index < len(totals) else 0,
        }
        if not stat or stat.get("mean") is None:
            cells.append({"na": True, **coverage})
            continue
        sig = False
        if best is not None and index != best and stats[best] is not None:
            sig = _significant_vs_best(
                metric, best, index, stats, runs_rows, confidence=confidence, resamples=resamples
            )
        cells.append(
            {
                "na": False,
                "mean": stat.get("mean"),
                "lo": stat.get("ci_low"),
                "hi": stat.get("ci_high"),
                "best": index == best,
                "worst": index == worst,
                "sig": bool(sig),
                **coverage,
            }
        )
    return cells


def _significant_vs_best(
    metric: str,
    best: int,
    index: int,
    stats: list[dict[str, Any] | None],
    runs_rows: list[list[dict[str, Any]]] | None,
    *,
    confidence: float,
    resamples: int,
) -> bool:
    """Star rule: the paired (per-text) difference to the best run excludes zero.

    Falls back to non-overlapping CIs when rows are unavailable or the runs share
    fewer than two texts (no pairing possible).
    """
    if runs_rows is not None and best < len(runs_rows) and index < len(runs_rows):
        interval = paired_difference_ci(
            _values_by_cluster(runs_rows[best], metric),
            _values_by_cluster(runs_rows[index], metric),
            confidence=confidence,
            resamples=resamples,
        )
        if interval is not None:
            low, high = interval
            return low > 0.0 or high < 0.0
    return intervals_separated(stats[best], stats[index])  # type: ignore[arg-type]


def _values_by_cluster(rows: list[dict[str, Any]], metric: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = metric_value(row, metric)
        if value is not None:
            grouped.setdefault(str(row.get(CLUSTER_KEY)), []).append(value)
    return grouped


def _health_table(
    all_rows: list[list[dict[str, Any]]], config: AssessmentConfig
) -> dict[str, Any]:
    good_band = config.reporting.health_good_rate
    warn_band = config.reporting.health_warn_rate
    # Only metrics with a configured threshold define a pass/fail, hence a "good %".
    metrics = [
        m
        for m in _ordered(set(config.thresholds))
        if m not in HEALTH_HIDE and _evaluated_anywhere(m, all_rows)
    ]
    rows = []
    for metric in metrics:
        cells = []
        for run_rows in all_rows:
            rate = _pass_rate(metric, run_rows)
            if rate is None:
                cells.append({"na": True})
            else:
                cells.append(
                    {
                        "na": False,
                        "status": _band(rate["good_rate"], good_band, warn_band),
                        **rate,
                    }
                )
        rows.append(
            {
                "metric": display_name(metric),
                "key": metric,
                "pass_rule": _pass_rule(config.thresholds[metric]),
                "cells": cells,
            }
        )
    return {"good_rate": good_band, "warn_rate": warn_band, "rows": rows}


def _pass_rule(threshold: Any) -> str:
    """Human-readable per-sample condition for a sample to count as passing."""
    if threshold.fail_if_true is not None:
        return "not flagged"
    # A sample passes when it is strictly better than the warn band (or the fail
    # band if no warn is set) in the metric's direction.
    if threshold.warn is not None or threshold.fail is not None:
        bound = threshold.warn if threshold.warn is not None else threshold.fail
        return f"< {_num(bound)}"
    if threshold.warn_below is not None or threshold.fail_below is not None:
        bound = threshold.warn_below if threshold.warn_below is not None else threshold.fail_below
        return f"> {_num(bound)}"
    return "—"


def _num(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".") or "0"


def _pass_rate(metric: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Share of clips passing ``metric``; failed measurements count as not passing."""
    evaluated = good = failed = 0
    for row in rows:
        if measurement_failure(row, metric) is not None:
            failed += 1
            continue
        status = (row.get("metric_statuses") or {}).get(metric)
        if status in ("pass", "warn", "fail"):
            evaluated += 1
            if status == "pass":
                good += 1
    if not evaluated and not failed:
        return None
    return {
        "good_rate": good / (evaluated + failed),
        "measured": evaluated,
        "failed": failed,
        "total": len(rows),
    }


def _evaluated_anywhere(metric: str, all_rows: list[list[dict[str, Any]]]) -> bool:
    return any(_pass_rate(metric, rows) is not None for rows in all_rows)


def _band(rate: float, good: float, warn: float) -> str:
    if rate >= good:
        return "good"
    if rate >= warn:
        return "warn"
    return "fail"


def _best_worst_stats(
    metric: str, stats: list[dict[str, Any] | None]
) -> tuple[int | None, int | None]:
    facing = direction(metric)
    means = {
        index: stat["mean"]
        for index, stat in enumerate(stats)
        if stat and stat.get("mean") is not None
    }
    if not facing or len(means) < 2:
        return None, None
    if facing == "lower":
        best = min(means, key=means.get)
        worst = max(means, key=means.get)
    else:
        best = max(means, key=means.get)
        worst = min(means, key=means.get)
    if means[best] == means[worst]:
        return None, None
    return best, worst


def _union_metrics(summaries: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for summary in summaries:
        names.update(summary.get("metrics", {}))
    return _ordered(names)


def _ordered(names: set[str]) -> list[str]:
    lead = [name for name in KEY_METRICS if name in names]
    rest = sorted(name for name in names if name not in KEY_METRICS)
    return lead + rest
