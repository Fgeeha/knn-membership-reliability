"""Selective-prediction metrics.

Tie-aware throughout: the retrieval head produces many exactly equal distances
(the ITIDS'2026 measurement reports a tie share up to 0.488 for C4-D), so any
metric that breaks ties by array order silently invents a signal that is not in
the data.
"""

from __future__ import annotations

import numpy as np

__all__ = ["auroc", "risk_coverage", "aurc", "ece", "bootstrap_ci"]

# numpy >= 2.0 renamed trapz to trapezoid; the plantdiag-edge venv has 1.26.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def auroc(scores: np.ndarray, correct: np.ndarray) -> float:
    """AUROC of `scores` predicting `correct`, with tied scores averaged.

    Args:
        scores:  Higher = predicted more likely correct.
        correct: 1 for a correct prediction, 0 otherwise.

    Returns:
        AUROC in [0, 1]; NaN when one of the two classes is empty.

    A value below 0.5 is meaningful, not a bug: it says the signal is
    anti-correlated with correctness. Callers must not take `max(a, 1 - a)`
    without choosing the direction on a separate tuning split.
    """
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.int64)
    if scores.shape != correct.shape:
        raise ValueError(f"shape mismatch: {scores.shape} vs {correct.shape}")

    n_pos = int(correct.sum())
    n_neg = correct.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = _average_ranks(scores[order])
    rank_sum_pos = float(ranks[correct[order] == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _average_ranks(sorted_values: np.ndarray) -> np.ndarray:
    """1-based ranks with ties replaced by their group mean."""
    n = sorted_values.size
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def risk_coverage(
    scores: np.ndarray, correct: np.ndarray, coverages: np.ndarray
) -> np.ndarray:
    """Accuracy of the highest-scoring fraction of queries, for each coverage.

    Ties at the cut-off are resolved by taking the whole tied group's mean
    accuracy weighted into the served set, which keeps the curve monotone in the
    absence of signal instead of jumping with array order.

    Args:
        scores:    Higher = served automatically first.
        correct:   1 for a correct prediction, 0 otherwise.
        coverages: Fractions in (0, 1].

    Returns:
        Array of the same length as `coverages`, holding accuracy on the served
        subset.
    """
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    n = scores.size
    order = np.argsort(-scores, kind="mergesort")
    ranked = correct[order]
    ranked_scores = scores[order]
    cum = np.cumsum(ranked)

    out = np.empty(len(coverages), dtype=np.float64)
    for idx, c in enumerate(coverages):
        m = max(1, int(round(c * n)))
        # Extend the cut to the end of the tied group, then rescale: the served
        # set is "m best" and members of a tied group are interchangeable.
        hi = m
        while hi < n and ranked_scores[hi] == ranked_scores[m - 1]:
            hi += 1
        lo = m - 1
        while lo > 0 and ranked_scores[lo - 1] == ranked_scores[m - 1]:
            lo -= 1
        if hi > lo:  # a tie straddles the cut
            head = cum[lo - 1] if lo > 0 else 0.0
            tie_acc = (cum[hi - 1] - head) / (hi - lo)
            out[idx] = (head + tie_acc * (m - lo)) / m
        else:
            out[idx] = cum[m - 1] / m
    return out


def aurc(scores: np.ndarray, correct: np.ndarray, n_points: int = 100) -> float:
    """Area under the risk-coverage curve (risk = 1 - accuracy). Lower is better."""
    coverages = np.linspace(1.0 / n_points, 1.0, n_points)
    acc = risk_coverage(scores, correct, coverages)
    return float(_trapezoid(1.0 - acc, coverages) / (coverages[-1] - coverages[0] + 1e-12))


def ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins on [0, 1].

    Only meaningful for a confidence that admits a probabilistic reading. The
    retrieval head's `1 / (1 + d)` does not, which is why ITIDS'2026 leaves the
    ECE column empty for those modes; callers should do the same rather than
    report a number that cannot be interpreted.
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence <= hi)
        if not mask.any():
            continue
        total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean. Deterministic given `seed`.

    Mirrors the convention already used by `plantdiag-edge/scripts/build_attribution.py`
    so intervals in this project are comparable with the published ones.
    """
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _demo() -> None:
    """Self-check; run as `python -m reliability_gate.metrics`."""
    rng = np.random.default_rng(0)

    # Perfect separation -> 1.0; reversed -> 0.0; constant -> 0.5.
    correct = np.array([0, 0, 1, 1])
    assert auroc(np.array([1.0, 2.0, 3.0, 4.0]), correct) == 1.0
    assert auroc(np.array([4.0, 3.0, 2.0, 1.0]), correct) == 0.0
    assert auroc(np.zeros(4), correct) == 0.5, "all ties must give exactly 0.5"

    # A signal that is 0.5 by construction stays 0.5 within sampling noise.
    c = rng.integers(0, 2, size=4000)
    assert abs(auroc(rng.normal(size=4000), c) - 0.5) < 0.03

    # Ranking-invariance: AUROC depends on order only.
    s = rng.normal(size=500)
    c2 = (rng.random(500) < 0.3).astype(int)
    assert abs(auroc(s, c2) - auroc(np.exp(s), c2)) < 1e-12

    # risk_coverage: full coverage is plain accuracy; a perfect signal is 1.0 early.
    s3 = np.array([0.9, 0.8, 0.1, 0.2])
    c3 = np.array([1, 1, 0, 0])
    assert abs(risk_coverage(s3, c3, np.array([1.0]))[0] - 0.5) < 1e-12
    assert abs(risk_coverage(s3, c3, np.array([0.5]))[0] - 1.0) < 1e-12

    # All-tied scores must not manufacture selectivity at any coverage.
    c4 = np.array([1, 0, 1, 0, 1, 0, 0, 0])
    for cov in (0.25, 0.5, 0.75, 1.0):
        got = risk_coverage(np.ones(8), c4, np.array([cov]))[0]
        assert abs(got - 0.375) < 1e-9, (cov, got)

    # AURC: a perfect signal beats a useless one.
    assert aurc(s3, c3) < aurc(np.ones(4), c3)

    # ECE: perfectly calibrated -> 0; fully overconfident -> 1.
    assert ece(np.full(100, 0.5), np.r_[np.ones(50), np.zeros(50)], n_bins=2) < 1e-9
    assert abs(ece(np.ones(100), np.zeros(100), n_bins=10) - 1.0) < 1e-9

    lo, hi = bootstrap_ci(np.array([0.60, 0.62, 0.58, 0.61, 0.59]))
    assert lo < 0.60 < hi and bootstrap_ci(np.array([0.60, 0.62, 0.58, 0.61, 0.59])) == (lo, hi)

    print("metrics self-check passed")


if __name__ == "__main__":
    _demo()
