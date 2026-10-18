#!/usr/bin/env python3
"""S3 step 3: primary endpoint, factor effects and the S2/S3 comparison.

The primary endpoint was fixed in `S3_PREREGISTRATION.md` before any S3 number
existed:

    delta = AUROC(correctness | matched gallery) - AUROC(correctness | mismatched)

for the built-in confidence signal, averaged over the remaining factors. It is
computed **paired by seed**, because every cell of a given seed sees the same
queries and the same gallery-subsampling draw; treating the arms as independent
would overstate the precision.

`S` has three levels, so the contrast is reported three ways: matched against the
two mismatched domains pooled (the endpoint), and matched against each of them
separately. The separate form is what distinguishes "lab versus field" from
"matched versus mismatched", which is the falsifiable part of the design.
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

from reliability_gate.metrics import bootstrap_ci  # noqa: E402

MATCHED = "plantwild"
MISMATCHED = ("plantvillage", "plantdoc")
SIGNALS_REPORT = ("neg_d1", "class_gap", "z_class", "margin_rel", "vote_share",
                  "neg_local_density", "d1_over_dbar", "rank_class",
                  "neg_vote_entropy", "z_times_vote")


def git_hash(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def agg(vals) -> dict:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"mean": None, "std": None, "n": 0}
    lo, hi = bootstrap_ci(v) if v.size > 1 else (float(v[0]), float(v[0]))
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "ci95": [lo, hi],
        "min": float(v.min()), "max": float(v.max()),
        "sign_stable": bool((v > 0).all() or (v < 0).all()),
        "n": int(v.size),
        "per_seed": v.tolist(),
    }


def signal_value(cell: dict, sig: str, field: str) -> float:
    s = cell["signals"].get(sig, {})
    if s.get("auroc") is None:
        return float("nan")
    return s.get(field, float("nan"))


def paired_contrast(cells, sig, field, arm_a, arm_b, arm_key="S") -> dict:
    """Per-seed difference between two levels, averaged over the other factors."""
    by_seed: dict[int, dict[str, list[float]]] = {}
    for c in cells:
        lvl = c["factors"][arm_key]
        if lvl not in (arm_a, arm_b) and not (isinstance(arm_b, tuple) and lvl in arm_b):
            continue
        side = "a" if lvl == arm_a else "b"
        by_seed.setdefault(c["seed"], {"a": [], "b": []})[side].append(
            signal_value(c, sig, field)
        )
    diffs = []
    for seed, d in sorted(by_seed.items()):
        if not d["a"] or not d["b"]:
            continue
        diffs.append(float(np.nanmean(d["a"]) - np.nanmean(d["b"])))
    return agg(diffs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=Path, default=PROJECT / "results" / "s3_factorial")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--arm", default="none", help="adaptation arm for the primary endpoint")
    ap.add_argument("--k-rule", default="fixed")
    args = ap.parse_args()
    out = args.out or args.cells / f"s3_summary_{args.arm}_{args.k_rule}.json"

    allc = [json.loads(p.read_text()) for p in sorted(args.cells.glob("*.json"))
            if not p.name.startswith("s3_summary")]
    cells = [c for c in allc if c["factors"]["A"] == args.arm and c["k_rule"] == args.k_rule]
    if not cells:
        print("no matching cells", file=sys.stderr)
        return 1

    # ── per-configuration aggregates ─────────────────────────────────────────
    per_config: dict[str, dict] = {}
    for c in cells:
        e = per_config.setdefault(c["config"], {"factors": c["factors"], "acc": [], "mr": [],
                                                "k": c["k"], "n_gallery": c["n_gallery"],
                                                "sig": {}, "corr": {}})
        e["acc"].append(c["accuracy"])
        e["mr"].append(c["macro_recall"])
        for s in SIGNALS_REPORT:
            d = e["sig"].setdefault(s, {"auroc": [], "und": [], "auprc": [], "aurc": [],
                                        "dir": [], "cov": {}})
            if c["signals"].get(s, {}).get("auroc") is None:
                continue
            d["auroc"].append(c["signals"][s]["auroc"])
            d["und"].append(c["signals"][s]["auroc_undirected"])
            d["auprc"].append(c["signals"][s]["auprc"])
            d["aurc"].append(c["signals"][s]["aurc"])
            d["dir"].append(c["signals"][s]["direction"])
            for k, v in c["signals"][s]["accuracy_at_coverage"].items():
                d["cov"].setdefault(k, []).append(v)
        for s, cd in c["signal_correlations"].items():
            for bn, v in cd.items():
                if v is not None:
                    e["corr"].setdefault(s, {}).setdefault(bn, []).append(v)

    summary = {}
    for name, e in per_config.items():
        summary[name] = {
            "factors": e["factors"], "k": e["k"], "n_gallery": e["n_gallery"],
            "accuracy": agg(e["acc"]), "macro_recall": agg(e["mr"]),
            "signals": {
                s: {
                    "auroc": agg(d["auroc"]), "auroc_undirected": agg(d["und"]),
                    "auprc": agg(d["auprc"]), "aurc": agg(d["aurc"]),
                    "direction_stable": len(set(d["dir"])) == 1,
                    "auroc_above_half_all_seeds": bool(all(a > 0.5 for a in d["auroc"])) if d["auroc"] else None,
                    "accuracy_at_coverage": {k: agg(v) for k, v in d["cov"].items()},
                }
                for s, d in e["sig"].items() if d["auroc"]
            },
            "spearman_vs": {s: {bn: agg(v) for bn, v in cd.items()} for s, cd in e["corr"].items()},
        }

    # ── primary endpoint and its decomposition ───────────────────────────────
    primary = {
        "definition": "AUROC(correctness | matched gallery) - AUROC(correctness | mismatched), "
                      "built-in confidence, paired by seed",
        "signal": "neg_d1",
        "field": "auroc_undirected",
        "matched_minus_mismatched_pooled": paired_contrast(
            cells, "neg_d1", "auroc_undirected", MATCHED, MISMATCHED),
        "matched_minus_plantvillage": paired_contrast(
            cells, "neg_d1", "auroc_undirected", MATCHED, "plantvillage"),
        "matched_minus_plantdoc": paired_contrast(
            cells, "neg_d1", "auroc_undirected", MATCHED, "plantdoc"),
        "plantdoc_minus_plantvillage": paired_contrast(
            cells, "neg_d1", "auroc_undirected", "plantdoc", "plantvillage"),
    }

    # ── secondary factor effects, same paired construction ───────────────────
    effects = {}
    for sig in ("neg_d1", "class_gap", "z_class"):
        field = "auroc_undirected" if sig == "neg_d1" else "auroc"
        effects[sig] = {
            "S_matched_minus_mismatched": paired_contrast(cells, sig, field, MATCHED, MISMATCHED),
            "G_pq3bit_minus_float32": paired_contrast(cells, sig, field, "pq3bit", "float32", "G"),
            "Q_int8_minus_fp32": paired_contrast(cells, sig, field, "onnx_int8", "onnx_fp32", "Q"),
        }

    # ── level means, for the inversion statement ─────────────────────────────
    by_level = {}
    for lvl in (MATCHED, *MISMATCHED):
        sel = [c for c in cells if c["factors"]["S"] == lvl]
        by_level[lvl] = {
            "accuracy": agg([c["accuracy"] for c in sel]),
            "n_cells": len(sel),
            "signals": {
                s: {
                    "auroc_undirected": agg([signal_value(c, s, "auroc_undirected") for c in sel]),
                    "auroc_directed": agg([signal_value(c, s, "auroc") for c in sel]),
                    "inverted": bool(np.nanmean([signal_value(c, s, "auroc_undirected") for c in sel]) < 0.5),
                }
                for s in SIGNALS_REPORT
            },
        }

    payload = {
        "arm": args.arm, "k_rule": args.k_rule,
        "primary_endpoint": primary,
        "factor_effects": effects,
        "by_gallery_domain": by_level,
        "per_config": summary,
        "meta": {
            "command": " ".join(sys.argv),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git": {"edge-reliability-gate": git_hash(PROJECT)},
            "n_cells": len(cells),
            "alignment": cells[0]["meta"]["alignment"],
            "environment": {"python": platform.python_version(), "numpy": np.__version__},
        },
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"arm A={args.arm}, k-rule {args.k_rule}, {len(cells)} cells\n")
    print(f"{'gallery domain':<16}{'cells':>6}{'acc':>9}{'neg_d1(und)':>13}{'inverted':>10}"
          f"{'class_gap':>11}{'z_class':>10}")
    for lvl in (MATCHED, *MISMATCHED):
        b = by_level[lvl]
        print(f"{lvl:<16}{b['n_cells']:>6}{b['accuracy']['mean']:>9.4f}"
              f"{b['signals']['neg_d1']['auroc_undirected']['mean']:>13.3f}"
              f"{str(b['signals']['neg_d1']['inverted']):>10}"
              f"{b['signals']['class_gap']['auroc_directed']['mean']:>11.3f}"
              f"{b['signals']['z_class']['auroc_directed']['mean']:>10.3f}")

    print("\nPRIMARY ENDPOINT (paired by seed):")
    for k in ("matched_minus_mismatched_pooled", "matched_minus_plantvillage",
              "matched_minus_plantdoc", "plantdoc_minus_plantvillage"):
        a = primary[k]
        print(f"  {k:<36} {a['mean']:+.4f} ± {a['std']:.4f}  "
              f"CI[{a['ci95'][0]:+.4f},{a['ci95'][1]:+.4f}]  "
              f"{'sign stable' if a['sign_stable'] else 'SIGN FLIPS'}  per-seed="
              f"{[round(x, 3) for x in a['per_seed']]}")

    print("\nFACTOR EFFECTS:")
    for sig, e in effects.items():
        print(f"  {sig}")
        for name, a in e.items():
            print(f"    {name:<32} {a['mean']:+.4f} ± {a['std']:.4f}  "
                  f"{'stable' if a['sign_stable'] else 'SIGN FLIPS'}")
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
