#!/usr/bin/env python3
"""S2 step 3: factor effects, interactions and the cause of confidence inversion.

Reads the per-cell JSON files written by `s2_run_factorial.py` and produces the
aggregate. Nothing here re-derives features or distances; it is pure aggregation,
so it can be re-run freely and `verify_s2_results.py` can recompute it
independently from the same inputs.

Effects use standard +-1 contrast coding over the 2^4 design. For a response y
measured at every one of the sixteen cells, the estimated effect of a term is

    effect(term) = mean(y where contrast = +1) - mean(y where contrast = -1)

which for a balanced two-level design equals twice the regression coefficient.
Interaction contrasts are elementwise products of their main-effect contrasts.

Three responses are analysed:
    inversion   AUROC of the built-in signal, *undirected* — below 0.5 means the
                signal is anti-correlated with correctness. Direction must not be
                repaired here, or the phenomenon under study disappears.
    class_gap   AUROC of the candidate, direction chosen on the tuning half.
    accuracy    top-1 accuracy of the cell, as a sanity response.
"""

from __future__ import annotations

import argparse
import itertools
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

FACTORS = ("A", "Q", "G", "S")
HIGH = {"A": "icoral", "Q": "onnx_int8", "G": "pq3bit", "S": "target"}
FACTOR_MEANING = {
    "A": "domain adaptation: none -> icoral",
    "Q": "backbone precision: onnx fp32 -> onnx int8 (real static QDQ)",
    "G": "gallery representation: float32 -> PolarQuant-3bit",
    "S": "gallery domain: source PlantVillage -> target PlantDoc-train",
}


def git_hash(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_cells(d: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json")) if p.name != "s2_summary.json"]


def agg(values: list[float]) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": None, "std": None, "ci95": None, "n": 0}
    lo, hi = bootstrap_ci(v) if v.size > 1 else (float(v[0]), float(v[0]))
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "ci95": [lo, hi],
        "min": float(v.min()),
        "max": float(v.max()),
        "n": int(v.size),
        "per_seed": v.tolist(),
    }


def response(cell: dict, which: str) -> float:
    if which == "accuracy":
        return cell["accuracy"]
    if which == "inversion":
        s = cell["signals"].get("neg_d1", {})
        return s.get("auroc_undirected", float("nan")) if s.get("auroc") is not None else float("nan")
    sig = cell["signals"].get(which, {})
    return sig.get("auroc", float("nan")) if sig.get("auroc") is not None else float("nan")


def factorial_effects(cells: list[dict], which: str) -> dict:
    """Main effects and all interactions of the 2^4 design for one response."""
    main = [c for c in cells if c["config"].startswith("s2_")]
    by_cell: dict[tuple, list[float]] = {}
    for c in main:
        key = tuple(c["factors"][f] for f in FACTORS)
        by_cell.setdefault(key, []).append(response(c, which))
    if len(by_cell) != 16:
        return {"error": f"expected 16 cells, found {len(by_cell)}"}

    keys = sorted(by_cell)
    y = np.array([np.nanmean(by_cell[k]) for k in keys])
    contrasts = {
        f: np.array([1.0 if k[i] == HIGH[f] else -1.0 for k in keys])
        for i, f in enumerate(FACTORS)
    }

    out: dict[str, dict] = {}
    for order in range(1, 5):
        for combo in itertools.combinations(FACTORS, order):
            c = np.ones(16)
            for f in combo:
                c = c * contrasts[f]
            name = "×".join(combo)
            # Per-seed effect estimates give the spread of the effect itself.
            per_seed = []
            n_seeds = min(len(v) for v in by_cell.values())
            for si in range(n_seeds):
                ys = np.array([by_cell[k][si] for k in keys])
                per_seed.append(float(ys[c > 0].mean() - ys[c < 0].mean()))
            ps = np.array(per_seed)
            lo, hi = bootstrap_ci(ps) if ps.size > 1 else (float(ps[0]), float(ps[0]))
            out[name] = {
                "order": order,
                "effect": float(y[c > 0].mean() - y[c < 0].mean()),
                "std_over_seeds": float(ps.std(ddof=1)) if ps.size > 1 else 0.0,
                "ci95": [lo, hi],
                "sign_stable": bool(np.all(ps > 0) or np.all(ps < 0)),
                "per_seed": per_seed,
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=Path, default=PROJECT / "results" / "s2_factorial")
    ap.add_argument("--out", type=Path, default=PROJECT / "results" / "s2_factorial" / "s2_summary.json")
    args = ap.parse_args()

    cells = load_cells(args.cells)
    if not cells:
        print("no cells found", file=sys.stderr)
        return 1
    signal_names = sorted({n for c in cells for n in c["signals"]})

    # ── per-configuration aggregate over seeds ───────────────────────────────
    per_config: dict[str, dict] = {}
    for c in cells:
        e = per_config.setdefault(
            c["config"],
            {"factors": c.get("factors"), "n_gallery": c["n_gallery"], "seeds": [],
             "accuracy": [], "macro_recall": [], "tie_share": [],
             "signals": {n: {"auroc": [], "auroc_undirected": [], "aurc": [], "aurc_random": [],
                             "direction": [], "cov": {}} for n in signal_names},
             "correlations": {n: {"neg_d1": [], "margin_rel": []} for n in signal_names}},
        )
        e["seeds"].append(c["seed"])
        e["accuracy"].append(c["accuracy"])
        e["macro_recall"].append(c["macro_recall"])
        e["tie_share"].append(c["tie_share_d1_d2"])
        for n in signal_names:
            s = c["signals"].get(n, {})
            if s.get("auroc") is None:
                continue
            e["signals"][n]["auroc"].append(s["auroc"])
            e["signals"][n]["auroc_undirected"].append(s["auroc_undirected"])
            e["signals"][n]["aurc"].append(s["aurc"])
            e["signals"][n]["aurc_random"].append(s["aurc_random"])
            e["signals"][n]["direction"].append(s["direction"])
            for cov, v in s["accuracy_at_coverage"].items():
                e["signals"][n]["cov"].setdefault(cov, []).append(v)
            e["signals"][n].setdefault("n_at_coverage", s["n_at_coverage"])
        for n, corr in c.get("signal_correlations", {}).items():
            for bn, v in corr.items():
                if v is not None:
                    e["correlations"][n][bn].append(v)

    summary: dict[str, dict] = {}
    for name, e in per_config.items():
        sigs = {}
        for n in signal_names:
            s = e["signals"][n]
            if not s["auroc"]:
                continue
            sigs[n] = {
                "auroc": agg(s["auroc"]),
                "auroc_undirected": agg(s["auroc_undirected"]),
                "aurc": agg(s["aurc"]),
                "aurc_random": agg(s["aurc_random"]),
                "direction_stable": len(set(s["direction"])) == 1,
                "auroc_above_half_all_seeds": bool(all(a > 0.5 for a in s["auroc"])),
                "accuracy_at_coverage": {k: agg(v) for k, v in s["cov"].items()},
                "n_at_coverage": s.get("n_at_coverage"),
                "spearman_vs": {
                    bn: agg(v) for bn, v in e["correlations"][n].items() if v
                },
            }
        summary[name] = {
            "factors": e["factors"],
            "n_gallery": e["n_gallery"],
            "seeds": e["seeds"],
            "accuracy": agg(e["accuracy"]),
            "macro_recall": agg(e["macro_recall"]),
            "tie_share_d1_d2": agg(e["tie_share"]),
            "signals": sigs,
        }

    # ── factor effects ───────────────────────────────────────────────────────
    responses = ["inversion", "accuracy", "class_gap", "vote_share", "z_class", "margin_rel"]
    effects = {r: factorial_effects(cells, r) for r in responses}

    # ── where is the built-in signal inverted? ───────────────────────────────
    inverted = {
        name: s["signals"]["neg_d1"]["auroc_undirected"]
        for name, s in summary.items()
        if "neg_d1" in s["signals"]
        and s["signals"]["neg_d1"]["auroc_undirected"]["mean"] is not None
        and s["signals"]["neg_d1"]["auroc_undirected"]["mean"] < 0.5
    }

    payload = {
        "per_config": summary,
        "factor_effects": effects,
        "factor_meaning": FACTOR_MEANING,
        "inverted_configs": {
            k: {"auroc_undirected_mean": v["mean"], "max_over_seeds": v["max"]}
            for k, v in inverted.items()
        },
        "meta": {
            "command": " ".join(sys.argv),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git": {"edge-reliability-gate": git_hash(PROJECT)},
            "n_cell_runs": len(cells),
            "n_configs": len(summary),
            "signals": signal_names,
            "environment": {"python": platform.python_version(), "numpy": np.__version__},
        },
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # ── console report ───────────────────────────────────────────────────────
    print(f"{len(cells)} cell-runs, {len(summary)} configurations\n")
    print(f"{'config':<24}{'N':>7}{'acc':>8}{'neg_d1(und)':>13}{'class_gap':>12}{'vote_share':>12}{'z_class':>10}")
    for name in sorted(summary):
        s = summary[name]
        def g(sig, field="auroc"):
            v = s["signals"].get(sig, {}).get(field, {}).get("mean")
            return f"{v:.3f}" if v is not None else "  -  "
        print(f"{name:<24}{s['n_gallery']:>7}{s['accuracy']['mean']:>8.4f}"
              f"{g('neg_d1', 'auroc_undirected'):>13}{g('class_gap'):>12}"
              f"{g('vote_share'):>12}{g('z_class'):>10}")

    print("\nconfigurations where the built-in signal is INVERTED (undirected AUROC < 0.5):")
    if inverted:
        for k, v in sorted(inverted.items(), key=lambda kv: kv[1]["mean"]):
            print(f"  {k:<24} {v['mean']:.3f}  (worst seed {v['min']:.3f}, best {v['max']:.3f})")
    else:
        print("  none")

    for r in ("inversion", "class_gap"):
        e = effects[r]
        if "error" in e:
            print(f"\neffects[{r}]: {e['error']}")
            continue
        print(f"\nfactor effects on {r} (sorted by |effect|):")
        for name, v in sorted(e.items(), key=lambda kv: -abs(kv[1]["effect"]))[:8]:
            flag = "stable" if v["sign_stable"] else "SIGN FLIPS"
            print(f"  {name:<12} {v['effect']:+.4f} ± {v['std_over_seeds']:.4f}  "
                  f"CI[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]  {flag}")

    print(f"\nwritten -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
