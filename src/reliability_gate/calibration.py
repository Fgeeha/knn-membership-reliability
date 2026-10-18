"""Reliability threshold with a finite-sample guarantee on the error rate. numpy only.

`TwoLoopGate.fit_quantile` serves a chosen fraction; practitioners usually need
the opposite: "among the served diagnoses, at most `target_risk` may be wrong".
This picks the loosest reliability threshold for which that holds with
probability at least `1 - delta` over the draw of the calibration set.

Procedure — Learn-then-Test (Angelopoulos et al., 2021) with a Bonferroni
correction: for each of `m` candidate thresholds, H0 "risk among served >
target" is tested with the exact binomial p-value P(Binom(n_served, target) <=
errors) at level `delta / m`; the loosest rejected threshold is returned.

Why not the more powerful fixed-sequence variant: it stops at the first failed
threshold, scanning from the strictest, and the strictest end of a compressed
retrieval score is noisy — on the C6 deploy profiles the top 1 % by `class_gap`
has risk 0.65 while 3-10 % has 0.39-0.52, so the scan died at n = 14 in 18 of
20 splits. Bonferroni needs no scan order and no knob; it costs coverage on
smooth curves (0.19 vs 0.27 in a simulation at target 0.2).

The guarantee assumes deployment queries are exchangeable with the calibration
queries. Under a domain shift — the very situation this project studies — it
does not hold; calibrate on data from the deployment domain.

Run:
    python -m reliability_gate.calibration calib.npz --target-risk 0.2 --out gate.json
where calib.npz holds `distances` (Q, N), `gallery_labels` (N,), `correct` (Q,).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from .gate import TwoLoopGate, scores, structural_k

__all__ = ["binom_cdf", "certify_threshold", "fit_gate_for_risk"]


def binom_cdf(x: int, n: int, p: float) -> float:
    """P(Binom(n, p) <= x), exact, in log space (stable for n in the thousands)."""
    if x >= n:
        return 1.0
    if x < 0:
        return 0.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    lf = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq for i in range(x + 1)]
    m = max(lf)
    return min(1.0, math.exp(m) * sum(math.exp(v - m) for v in lf))


def certify_threshold(
    reliability: np.ndarray,
    correct: np.ndarray,
    target_risk: float,
    delta: float = 0.10,
    n_grid: int = 100,
) -> dict:
    """Loosest threshold whose served error rate is certified <= `target_risk`.

    Args:
        reliability: (Q,) reliability scores of calibration queries; higher = trusted.
        correct:     (Q,) 1 where the prediction was correct.
        target_risk: Allowed error rate among served queries, in (0, 1).
        delta:       Allowed probability that the guarantee fails.
        n_grid:      Candidate thresholds, taken at quantiles of `reliability`.

    Returns:
        Dict with `certified`, `threshold` (inf when nothing is certified, i.e.
        serve nothing), and the calibration-set coverage/risk at it.
    """
    if not 0.0 < target_risk < 1.0:
        raise ValueError(f"target_risk must be in (0, 1), got {target_risk}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    rel = np.asarray(reliability, dtype=np.float64)
    wrong = 1 - np.asarray(correct, dtype=np.int64)
    # Unique, so ties never create two tests of the same served set.
    grid = np.unique(np.quantile(rel, np.linspace(0.0, 1.0, n_grid + 1)))
    level = delta / grid.size  # Bonferroni over every threshold tested

    best = None
    for thr in grid:  # loosest first: the first rejection is the answer
        served = rel >= thr
        n, e = int(served.sum()), int(wrong[served].sum())
        if binom_cdf(e, n, target_risk) <= level:
            best = (float(thr), n, e)
            break

    out = {"target_risk": target_risk, "delta": delta, "n_calibration": int(rel.size)}
    if best is None:
        return {**out, "certified": False, "threshold": float("inf"), "coverage": 0.0,
                "empirical_risk": float("nan"), "n_served": 0}
    thr, n, e = best
    return {**out, "certified": True, "threshold": thr, "coverage": n / rel.size,
            "empirical_risk": e / n, "n_served": n, "n_thresholds_tested": int(grid.size)}


def fit_gate_for_risk(
    distances: np.ndarray,
    gallery_labels: np.ndarray,
    correct: np.ndarray,
    target_risk: float,
    delta: float = 0.10,
    keep_in_space: float | None = 0.90,
    k: int | None = None,
) -> tuple[TwoLoopGate, dict]:
    """Two-loop gate whose reliability threshold carries the risk guarantee.

    The membership threshold is fitted as in `TwoLoopGate.fit_quantile`; the
    reliability threshold is certified on the queries that membership admits,
    so the guarantee is on the error rate of what the full gate serves.
    """
    k = structural_k(gallery_labels) if k is None else k
    _, memb, rel = scores(distances, gallery_labels, k)
    correct = np.asarray(correct)
    thr_memb, admitted = None, np.ones(rel.size, dtype=bool)
    if keep_in_space is not None:
        thr_memb = float(np.quantile(memb, 1.0 - keep_in_space))
        admitted = memb >= thr_memb
    rep = certify_threshold(rel[admitted], correct[admitted], target_risk, delta)
    rep["coverage_of_all"] = rep["n_served"] / rel.size
    gate = TwoLoopGate(k=k, reliability_threshold=rep["threshold"], membership_threshold=thr_memb)
    return gate, rep


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("npz", type=Path, help="calibration set: distances, gallery_labels, correct")
    p.add_argument("--target-risk", type=float, required=True)
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--keep-in-space", type=float, default=0.90, help="<= 0 disables the membership loop")
    p.add_argument("--out", type=Path, required=True, help="where to write the gate JSON")
    args = p.parse_args(argv)

    try:
        z = np.load(args.npz)
        keep = args.keep_in_space if args.keep_in_space > 0 else None
        gate, rep = fit_gate_for_risk(z["distances"], z["gallery_labels"], z["correct"],
                                      args.target_risk, args.delta, keep)
    except (OSError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    gate.save(args.out)
    print(json.dumps(rep, indent=2))
    if not rep["certified"]:
        print("no threshold certified: the gate will serve nothing (target too strict or too few queries)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
