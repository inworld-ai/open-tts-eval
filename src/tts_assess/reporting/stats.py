from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import numpy as np

# Fixed seed so a given set of values always yields the same interval; reports
# must be reproducible from the same inputs.
_BOOTSTRAP_SEED = 20240501


class Interval(TypedDict):
    n: int
    mean: float | None
    std: float | None
    median: float | None
    p95: float | None
    min: float | None
    max: float | None
    ci_low: float | None
    ci_high: float | None
    ci_level: float
    ci_method: str


def summarize_values(
    values: list[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    clusters: Sequence[str] | None = None,
) -> Interval:
    """Central tendency, dispersion, and a bootstrap CI for the mean.

    ``clusters`` (one label per value, e.g. the source text) makes the CI a
    cluster bootstrap: clips that share a text are not independent draws.
    """
    if not values:
        return Interval(
            n=0,
            mean=None,
            std=None,
            median=None,
            p95=None,
            min=None,
            max=None,
            ci_low=None,
            ci_high=None,
            ci_level=confidence,
            ci_method="none",
        )
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    ci_low, ci_high = bootstrap_ci(
        arr, confidence=confidence, resamples=resamples, clusters=clusters
    )
    return Interval(
        n=int(arr.size),
        mean=mean,
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        median=float(np.median(arr)),
        p95=float(np.percentile(arr, 95)),
        min=float(arr.min()),
        max=float(arr.max()),
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=confidence,
        ci_method=bootstrap_method(arr.size, clusters),
    )


def bootstrap_method(n: int, clusters: Sequence[str] | None) -> str:
    """Which bootstrap applies: ``cluster`` when there are 2+ clusters smaller than n."""
    if clusters is not None and 2 <= len(set(clusters)) < n:
        return "cluster"
    return "iid"


def bootstrap_ci(
    values: np.ndarray | list[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    clusters: Sequence[str] | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    Non-parametric: makes no normality assumption, which suits skewed and
    bounded TTS metrics (WER, similarity, silence ratio). With ``clusters``,
    whole clusters are resampled (the pooled mean over the drawn clusters), so
    shared text difficulty across voices widens the interval as it should. With
    fewer than two samples the interval collapses to the point estimate.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    if bootstrap_method(arr.size, clusters) == "cluster":
        means = _cluster_means(arr, list(clusters), resamples, rng)  # type: ignore[arg-type]
    else:
        samples = rng.choice(arr, size=(resamples, arr.size), replace=True)
        means = samples.mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low = float(np.percentile(means, 100 * alpha))
    high = float(np.percentile(means, 100 * (1.0 - alpha)))
    return (low, high)


def _cluster_means(
    arr: np.ndarray, clusters: list[str], resamples: int, rng: np.random.Generator
) -> np.ndarray:
    labels, inverse = np.unique(np.asarray(clusters, dtype=object), return_inverse=True)
    sums = np.bincount(inverse, weights=arr, minlength=len(labels))
    counts = np.bincount(inverse, minlength=len(labels)).astype(float)
    idx = rng.integers(0, len(labels), size=(resamples, len(labels)))
    return sums[idx].sum(axis=1) / counts[idx].sum(axis=1)


def paired_difference_ci(
    a_by_cluster: dict[str, Sequence[float]],
    b_by_cluster: dict[str, Sequence[float]],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
) -> tuple[float, float] | None:
    """Bootstrap CI of the mean paired difference (a - b) over shared clusters.

    Each shared cluster (text) contributes the difference of its per-run means,
    so voice-level noise and text difficulty cancel. Returns None when fewer
    than two clusters are shared (no pairing possible).
    """
    shared = sorted(set(a_by_cluster) & set(b_by_cluster))
    diffs = np.asarray(
        [
            float(np.mean(a_by_cluster[key])) - float(np.mean(b_by_cluster[key]))
            for key in shared
            if len(a_by_cluster[key]) and len(b_by_cluster[key])
        ],
        dtype=float,
    )
    if diffs.size < 2:
        return None
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    samples = rng.choice(diffs, size=(resamples, diffs.size), replace=True)
    means = samples.mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.percentile(means, 100 * alpha)),
        float(np.percentile(means, 100 * (1.0 - alpha))),
    )


def intervals_separated(a: Interval, b: Interval) -> bool:
    """True when two CIs do not overlap (a conservative significance signal)."""
    if a["ci_low"] is None or b["ci_low"] is None:
        return False
    return a["ci_high"] < b["ci_low"] or b["ci_high"] < a["ci_low"]
