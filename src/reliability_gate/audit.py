"""Audit a classifier's confidence signal: informative, uninformative or inverted?

The question this answers is the one the ITIDS'2026 measurement raised for the
C6 pipeline: does "serve the most confident queries first" actually select the
queries the model gets right? The input is any classifier's per-query dump —
a confidence score and whether the prediction was correct — so the audit is not
tied to this project's pipeline.

Run:
    python -m reliability_gate.audit dump.csv --score conf --correct correct
    python -m reliability_gate.audit dump.csv --score conf --correct correct --json out.json

The verdict comes from a percentile bootstrap CI of the AUROC over queries:
`inverted` when the whole interval lies below 0.5, `informative` when it lies
above, `uninformative` otherwise. No direction is ever flipped: an AUROC below
0.5 is reported as it is, because flipping it on the same data would turn noise
into an apparent signal (see `test_no_signal_in_pure_noise`).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .metrics import auroc, risk_coverage

__all__ = ["audit", "format_report"]

COVERAGES = (1.0, 0.5, 0.25, 0.10)


def _auroc_ci(scores: np.ndarray, correct: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """Percentile bootstrap CI of the AUROC, resampling queries."""
    rng = np.random.default_rng(seed)
    n = scores.size
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a = auroc(scores[idx], correct[idx])
        if np.isfinite(a):
            vals.append(a)
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def audit(
    scores: np.ndarray,
    correct: np.ndarray,
    coverages: tuple[float, ...] = COVERAGES,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Audit one confidence signal.

    Args:
        scores:    (N,) confidence; higher = the model claims more certainty.
        correct:   (N,) 1 where the prediction was correct, 0 otherwise.
        coverages: Served fractions at which to report accuracy.
        n_boot:    Bootstrap resamples for the AUROC interval.
        seed:      Bootstrap seed; the result is deterministic given it.

    Returns:
        Dict with the AUROC, its 95 % CI, the verdict, accuracy at each coverage
        and in the top/bottom confidence deciles.
    """
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.int64)
    if scores.shape != correct.shape or scores.ndim != 1:
        raise ValueError(f"expected two equal 1-D arrays, got {scores.shape} and {correct.shape}")
    if not np.isin(correct, (0, 1)).all():
        raise ValueError("`correct` must hold only 0 and 1")
    finite = np.isfinite(scores)
    if not finite.all():
        raise ValueError(f"{int((~finite).sum())} non-finite scores; drop or impute them first")

    a = auroc(scores, correct)
    lo, hi = _auroc_ci(scores, correct, n_boot, seed)
    if hi < 0.5:
        verdict = "inverted"
    elif lo > 0.5:
        verdict = "informative"
    else:
        verdict = "uninformative"

    decile = risk_coverage(scores, correct, np.array([0.10]))[0]
    bottom = risk_coverage(-scores, correct, np.array([0.10]))[0]
    return {
        "n": int(scores.size),
        "accuracy": float(correct.mean()),
        "auroc": a,
        "auroc_ci95": [lo, hi],
        "verdict": verdict,
        "accuracy_at_coverage": {
            str(c): float(v) for c, v in zip(coverages, risk_coverage(scores, correct, np.array(coverages)))
        },
        "accuracy_top_decile": float(decile),
        "accuracy_bottom_decile": float(bottom),
        "n_boot": n_boot,
        "seed": seed,
    }


_EXPLAIN = {
    "inverted": "serving the most confident queries first selects MORE errors than chance",
    "uninformative": "confidence does not separate correct from wrong predictions",
    "informative": "confidence ranks correct predictions above wrong ones",
}


def format_report(r: dict) -> str:
    """Human-readable form of `audit()` output."""
    lo, hi = r["auroc_ci95"]
    lines = [
        f"queries: {r['n']}   accuracy: {r['accuracy']:.3f}",
        f"AUROC:   {r['auroc']:.3f}  (95% CI {lo:.3f}-{hi:.3f}, {r['n_boot']} bootstrap resamples)",
        f"verdict: {r['verdict'].upper()} - {_EXPLAIN[r['verdict']]}",
        "",
        "accuracy on the served (most confident) fraction:",
    ]
    lines += [f"  coverage {float(c):>5.0%}: {v:.3f}" for c, v in r["accuracy_at_coverage"].items()]
    lines += [
        f"top decile {r['accuracy_top_decile']:.3f} vs bottom decile {r['accuracy_bottom_decile']:.3f}",
    ]
    return "\n".join(lines)


def _read_csv(path: Path, score_col: str, correct_col: str) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: no rows")
    for col in (score_col, correct_col):
        if col not in rows[0]:
            raise ValueError(f"{path}: no column {col!r}; have {sorted(rows[0])}")
    scores = np.array([float(r[score_col]) for r in rows])
    correct = np.array([int(float(r[correct_col])) for r in rows])
    return scores, correct


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("csv", type=Path, help="per-query dump with a header row")
    p.add_argument("--score", required=True, help="column with the confidence score")
    p.add_argument("--correct", required=True, help="column with 1 (correct) / 0 (wrong)")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", type=Path, help="also write the result as JSON")
    args = p.parse_args(argv)

    try:
        scores, correct = _read_csv(args.csv, args.score, args.correct)
        r = audit(scores, correct, n_boot=args.n_boot, seed=args.seed)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_report(r))
    if args.json:
        args.json.write_text(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
