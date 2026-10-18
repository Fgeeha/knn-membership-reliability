#!/usr/bin/env python3
"""Probe: can the published per-sample dumps alone fix the confidence signal?

This is the cheapest possible attack on the problem and it runs before any new
computation, precisely so that the expensive path is only taken if the cheap one
fails. It re-reads the per-sample dumps that `plantdiag-edge` already wrote for
the factorial plan and asks whether any rearrangement of the fields recorded
there (`conf`, `d1`, `d2`, `margin`) recovers a usable selection signal for the
retrieval modes.

It answers two questions:

1. Is the pipeline's `conf` anything other than a monotone function of `d1`?
   (If not, ITIDS'2026's `conf` column and a raw-distance column are the same
   measurement under two names.)
2. Does any simple rescoring reach AUROC above chance?

No new inference is performed and nothing is written into `plantdiag-edge`.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from reliability_gate.metrics import auroc  # noqa: E402

DEFAULT_DUMPS = PROJECT.parent / "plantdiag-edge" / "results"
RETRIEVAL_MODES = (
    "c6_c_fp32_icoral_pq_knn",
    "c6_d_int8_icoral_pq_knn",
    "c4_c_fp32_icoral_pq_knn",
    "c4_d_int8_icoral_pq_knn",
)
LINEAR_MODES = ("c6_a_fp32_icoral_linear", "c4_a_fp32_icoral_linear")


def git_hash(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def derived_signals(rows: list[dict]) -> dict[str, np.ndarray]:
    """Every signal expressible from the recorded fields, oriented higher=better."""
    d1 = np.array([r["d1"] for r in rows], dtype=np.float64)
    d2 = np.array([r["d2"] for r in rows], dtype=np.float64)
    out = {
        "conf": np.array([r["conf"] for r in rows], dtype=np.float64),
        "margin": np.array([r["margin"] for r in rows], dtype=np.float64),
        "neg_d1": -d1,
        "d2_minus_d1": d2 - d1,
        "d1_over_d2": d1 / np.maximum(d2, 1e-12),
        "neg_d1_plus_d2": -(d1 + d2),
    }
    return out


def linear_signals(rows: list[dict]) -> dict[str, np.ndarray]:
    p1 = np.array([r["p1"] for r in rows], dtype=np.float64)
    p2 = np.array([r["p2"] for r in rows], dtype=np.float64)
    return {"p1": p1, "margin": p1 - p2}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", type=Path, default=DEFAULT_DUMPS)
    ap.add_argument("--out", type=Path, default=PROJECT / "results" / "probe_baseline_signals")
    args = ap.parse_args()

    run_dirs = sorted(args.dumps.glob("c4_c6_ablation_seed*_split42_persample_*"))
    if not run_dirs:
        print(f"no per-sample dumps under {args.dumps}", file=sys.stderr)
        return 1

    report: dict[str, dict] = {}
    for mode in RETRIEVAL_MODES + LINEAR_MODES:
        is_linear = mode in LINEAR_MODES
        acc, by_signal = [], {}
        conf_matches_negd1 = []
        for d in run_dirs:
            rows = [json.loads(line) for line in (d / "per_sample" / f"{mode}.jsonl").open()]
            correct = np.array([r["correct"] for r in rows], dtype=np.int64)
            acc.append(float(correct.mean()))
            sig = linear_signals(rows) if is_linear else derived_signals(rows)
            for name, s in sig.items():
                by_signal.setdefault(name, []).append(auroc(s, correct))
            if not is_linear:
                conf_matches_negd1.append(
                    abs(auroc(sig["conf"], correct) - auroc(sig["neg_d1"], correct)) < 1e-9
                )
        report[mode] = {
            "n_seeds": len(run_dirs),
            "n_samples": len(rows),
            "accuracy": {"mean": float(np.mean(acc)), "std": float(np.std(acc, ddof=1))},
            "auroc": {
                name: {
                    "mean": float(np.mean(v)),
                    "std": float(np.std(v, ddof=1)),
                    "per_seed": [float(x) for x in v],
                }
                for name, v in by_signal.items()
            },
        }
        if not is_linear:
            report[mode]["conf_is_monotone_in_neg_d1"] = bool(all(conf_matches_negd1))

    best_retrieval = max(
        report[m]["auroc"][s]["mean"]
        for m in RETRIEVAL_MODES
        for s in report[m]["auroc"]
    )
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "conclusion": {
            "best_auroc_over_all_retrieval_signals": best_retrieval,
            "cheap_fix_exists": bool(best_retrieval >= 0.60),
            "threshold_note": (
                "0.60 is the success criterion recorded in NEXT_RESEARCH_DECISION.md "
                "before this probe was run"
            ),
        },
        "modes": report,
        "meta": {
            "command": " ".join(sys.argv),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_runs": [str(d) for d in run_dirs],
            "git": {
                "edge-reliability-gate": git_hash(PROJECT),
                "plantdiag-edge": git_hash(args.dumps.parent),
            },
            "environment": {"python": platform.python_version(), "numpy": np.__version__},
        },
    }
    (args.out / "probe_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    for mode in RETRIEVAL_MODES + LINEAR_MODES:
        r = report[mode]
        flag = r.get("conf_is_monotone_in_neg_d1")
        note = "" if flag is None else f"   conf == f(-d1): {flag}"
        print(f"\n{mode}  accuracy {r['accuracy']['mean']:.3f}{note}")
        for name, a in sorted(r["auroc"].items(), key=lambda kv: -kv[1]["mean"]):
            print(f"   {name:16s} AUROC = {a['mean']:.3f} ± {a['std']:.3f}")
    print(
        f"\nbest AUROC over every retrieval signal: {best_retrieval:.3f} -> "
        f"cheap fix {'EXISTS' if payload['conclusion']['cheap_fix_exists'] else 'DOES NOT EXIST'}"
    )
    print(f"written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
