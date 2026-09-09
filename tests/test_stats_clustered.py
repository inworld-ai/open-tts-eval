"""Confidence intervals respect the text x voice design (issue #3, H2)."""

import numpy as np
import pytest

from tts_assess.config import AssessmentConfig
from tts_assess.reporting.aggregate import summarize
from tts_assess.reporting.compare import build_comparison
from tts_assess.reporting.stats import (
    bootstrap_ci,
    bootstrap_method,
    paired_difference_ci,
    summarize_values,
)


def _grid(n_texts: int, n_voices: int, *, text_effect: float, noise: float, seed: int):
    """values[i] and cluster labels for a texts x voices grid with shared text difficulty."""
    rng = np.random.default_rng(seed)
    difficulty = rng.normal(0.0, text_effect, n_texts)
    values, clusters = [], []
    for t in range(n_texts):
        for _v in range(n_voices):
            values.append(0.1 + difficulty[t] + rng.normal(0.0, noise))
            clusters.append(f"text{t}")
    return values, clusters


def test_bootstrap_method_selection():
    assert bootstrap_method(10, None) == "iid"
    assert bootstrap_method(10, ["a"] * 10) == "iid"  # one cluster: nothing to resample
    assert bootstrap_method(10, [str(i) for i in range(10)]) == "iid"  # all distinct
    assert bootstrap_method(10, ["a", "b"] * 5) == "cluster"


def test_cluster_bootstrap_is_wider_when_texts_share_difficulty():
    values, clusters = _grid(50, 10, text_effect=0.05, noise=0.01, seed=0)
    iid_low, iid_high = bootstrap_ci(values, resamples=1000)
    cl_low, cl_high = bootstrap_ci(values, resamples=1000, clusters=clusters)
    mean = float(np.mean(values))
    assert cl_low <= mean <= cl_high
    assert (cl_high - cl_low) > 1.5 * (iid_high - iid_low)
    # Deterministic.
    assert bootstrap_ci(values, resamples=1000, clusters=clusters) == (cl_low, cl_high)


def test_cluster_bootstrap_matches_iid_when_texts_carry_no_effect():
    values, clusters = _grid(50, 10, text_effect=0.0, noise=0.02, seed=1)
    iid_low, iid_high = bootstrap_ci(values, resamples=2000)
    cl_low, cl_high = bootstrap_ci(values, resamples=2000, clusters=clusters)
    assert abs((cl_high - cl_low) - (iid_high - iid_low)) < 0.3 * (iid_high - iid_low)


def test_summarize_reports_ci_method_and_uses_text_clusters():
    rows = []
    values, clusters = _grid(20, 5, text_effect=0.1, noise=0.01, seed=2)
    for i, (value, text) in enumerate(zip(values, clusters, strict=True)):
        rows.append({"id": str(i), "text": text, "speaker_id": f"v{i % 5}", "wer": value})
    summary = summarize(rows, bootstrap_resamples=500, group_by_voice=False)
    assert summary["metrics"]["wer"]["count"] == 100
    method = summarize_values(values, resamples=10, clusters=clusters)["ci_method"]
    assert method == "cluster"
    # Rows with all-distinct texts fall back to iid.
    single = [{"id": str(i), "text": f"t{i}", "wer": v} for i, v in enumerate(values)]
    assert summarize_values([r["wer"] for r in single], resamples=10,
                            clusters=[r["text"] for r in single])["ci_method"] == "iid"


def test_paired_difference_ci():
    a = {f"t{i}": [0.10 + 0.01 * i, 0.12 + 0.01 * i] for i in range(20)}
    b = {key: [v + 0.002 for v in vals] for key, vals in a.items()}
    low, high = paired_difference_ci(a, b, resamples=500)
    assert low == pytest.approx(-0.002) and high == pytest.approx(-0.002)
    assert high < 0.0  # b is consistently higher, so the interval excludes zero
    assert paired_difference_ci({"t0": [1.0]}, {"t0": [2.0]}) is None  # one cluster
    assert paired_difference_ci(a, {"other": [1.0]}) is None  # nothing shared


def _runs(delta_per_text, noise_seed):
    rng = np.random.default_rng(noise_seed)
    a, b = [], []
    for t in range(30):
        base = 0.05 + 0.3 * (t / 30)  # texts vary a lot in difficulty
        for v in range(2):
            a.append({"id": f"a_{t}_{v}", "speaker_id": f"v{v}", "text": f"text {t}", "wer": base})
            b.append({
                "id": f"b_{t}_{v}", "speaker_id": f"v{v}", "text": f"text {t}",
                "wer": base + delta_per_text(t, rng),
            })
    return [("a", a), ("b", b)]


def _wer_row(report):
    for group in report["metric_table"]["groups"]:
        for row in group["rows"]:
            if row["key"] == "wer":
                return row
    raise KeyError("wer")


def test_star_uses_paired_per_text_differences():
    config = AssessmentConfig.default()
    config.reporting.bootstrap_resamples = 500

    # A small but perfectly consistent per-text degradation: run CIs overlap
    # (text difficulty dominates), yet the paired difference is unmistakable.
    consistent = build_comparison(_runs(lambda t, rng: 0.004, 0), config)
    cells = _wer_row(consistent)["cells"]
    assert cells[0]["best"] is True and cells[1]["sig"] is True
    assert cells[0]["hi"] > cells[1]["lo"]  # intervals overlap: the old rule would not star

    # Per-text differences that average to nothing must not earn a star.
    noisy = build_comparison(_runs(lambda t, rng: rng.normal(0.0, 0.02), 1), config)
    assert all(cell["sig"] is False for cell in _wer_row(noisy)["cells"])
    assert consistent["metric_table"]["ci_method"] == "cluster"
