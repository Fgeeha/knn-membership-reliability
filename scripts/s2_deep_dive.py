#!/usr/bin/env python3
"""S2 step 4: k-ablation, full risk-coverage curves, per-class analysis, failures.

Four analyses, all of them declared in `plan/RESEARCH_PLAN.md` or
`FACTORIAL_DESIGN.md` before the factorial was run:

  **k-ablation** (ablation 1 of the research plan). The factorial fixes k = 10.
  At the matched gallery size that is roughly a whole class; at the full
  32 605-vector source gallery it is under 1 % of one. Whether `class_gap`
  fails at the full gallery *because of the gallery* or *because of k* is a
  question about a declared hyperparameter, not a search for a better number,
  and both outcomes are reported.

  **Risk-coverage curves** over the whole coverage range rather than four
  convenient points, with the surviving sample size printed at every point.

  **Per-class analysis** — whether an aggregate AUROC is carried by a handful of
  easy classes.

  **Failure cases** — the four corners of (signal high/low) x (correct/wrong).

Reuses `build_cell_arrays` from `s2_run_factorial`, so the cells analysed here
are constructed by exactly the same code path the factorial used.
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
PLANTDIAG = PROJECT.parent / "plantdiag-edge"
sys.path.insert(0, str(PLANTDIAG))
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from reliability_gate.metrics import aurc, auroc, bootstrap_ci, risk_coverage  # noqa: E402
from reliability_gate.signals import compute_all  # noqa: E402
from s2_run_factorial import build_cell_arrays, build_configs  # noqa: E402

K_GRID = (1, 3, 5, 10, 20, 50, 100, 200, 500)
CURVE = np.round(np.arange(0.05, 1.0001, 0.05), 2)
SEEDS = (0, 1, 2, 3, 42)


def git_hash(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def split_halves(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Exactly the split the factorial runner uses, so results stay comparable."""
    idx = np.random.default_rng(seed).permutation(n)
    return idx[: n // 2], idx[n // 2 :]


def directed_eval(s: np.ndarray, correct: np.ndarray, seed: int) -> dict | None:
    finite = np.isfinite(s)
    tune_all, eval_all = split_halves(correct.size, seed)
    tune, ev = tune_all[finite[tune_all]], eval_all[finite[eval_all]]
    if tune.size == 0 or ev.size == 0 or correct[ev].sum() in (0, ev.size):
        return None
    a_tune = auroc(s[tune], correct[tune])
    direction = 1.0 if (np.isnan(a_tune) or a_tune >= 0.5) else -1.0
    se, ce = direction * s[ev], correct[ev]
    return {
        "auroc": auroc(se, ce),
        "auroc_undirected": auroc(s[ev], ce),
        "direction": direction,
        "aurc": aurc(se, ce),
        "n_eval": int(ev.size),
        "curve": risk_coverage(se, ce, CURVE).tolist(),
        "curve_n": [int(max(1, round(c * ce.size))) for c in CURVE],
        "base_accuracy": float(ce.mean()),
    }


def agg(v: list[float]) -> dict:
    a = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if a.size == 0:
        return {"mean": None, "std": None}
    lo, hi = bootstrap_ci(a) if a.size > 1 else (float(a[0]), float(a[0]))
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "ci95": [lo, hi],
        "min": float(a.min()),
        "max": float(a.max()),
        "all_above_half": bool((a > 0.5).all()),
        "per_seed": a.tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=PROJECT / "results" / "s2_features")
    ap.add_argument("--out", type=Path, default=PROJECT / "results" / "s2_deep_dive")
    ap.add_argument(
        "--configs", nargs="+",
        default=["s2_1010", "ref_fullsrc_fp32_pq", "ref_torch_pq_src", "s2_1011"],
    )
    ap.add_argument("--signals", nargs="+", default=["class_gap", "vote_share", "neg_d1", "z_class", "margin_rel"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--k-grid", type=int, nargs="+", default=list(K_GRID),
                    help="neighbourhood sizes for the declared k-ablation")
    args = ap.parse_args()
    k_grid = tuple(args.k_grid)

    args.out.mkdir(parents=True, exist_ok=True)
    cache = {p.stem: np.load(p) for p in args.features.glob("*.npy")}
    by_name = {c["name"]: c for c in build_configs()}
    payload: dict = {"k_ablation": {}, "curves": {}, "per_class": {}, "failures": {}}

    for cname in args.configs:
        cfg = by_name[cname]
        print(f"\n=== {cname}  (A={cfg['A']} Q={cfg['Q']} G={cfg['G']} S={cfg['S']}"
              f"{' full-source' if cfg.get('full_source') else ''}) ===", flush=True)

        built_by_seed = {}
        for seed in args.seeds:
            built_by_seed[seed] = build_cell_arrays(cfg, cache, seed)

        n_gal = len(built_by_seed[args.seeds[0]]["gallery_labels"])
        per_class_size = n_gal / built_by_seed[args.seeds[0]]["n_classes"]

        # ── k-ablation ───────────────────────────────────────────────────────
        ka: dict = {}
        for k in k_grid:
            if k > n_gal:
                continue
            for seed, b in built_by_seed.items():
                scores, pred = compute_all(b["dists"], b["gallery_labels"], b["stats"], k=k)
                correct = (pred == b["query_labels"]).astype(np.int64)
                for sig in args.signals:
                    r = directed_eval(scores[sig], correct, seed)
                    if r is None:
                        continue
                    ka.setdefault(sig, {}).setdefault(str(k), []).append(r["auroc"])
        payload["k_ablation"][cname] = {
            "n_gallery": n_gal,
            "mean_entries_per_class": per_class_size,
            "by_signal": {s: {k: agg(v) for k, v in d.items()} for s, d in ka.items()},
        }
        print(f"  gallery {n_gal} vectors, ~{per_class_size:.0f} per class")
        print(f"  {'k':>5} " + "".join(f"{s:>13}" for s in args.signals))
        for k in k_grid:
            if k > n_gal:
                continue
            row = f"  {k:>5} "
            for s in args.signals:
                m = payload["k_ablation"][cname]["by_signal"].get(s, {}).get(str(k), {}).get("mean")
                row += f"{m:>13.3f}" if m is not None else f"{'-':>13}"
            print(row)

        # ── curves, per-class, failures at the factorial's k = 10 ────────────
        best_k = 10
        curves: dict = {}
        pc: dict = {}
        for seed, b in built_by_seed.items():
            scores, pred = compute_all(b["dists"], b["gallery_labels"], b["stats"], k=best_k)
            correct = (pred == b["query_labels"]).astype(np.int64)
            q_lbl = b["query_labels"]
            for sig in args.signals:
                r = directed_eval(scores[sig], correct, seed)
                if r is None:
                    continue
                curves.setdefault(sig, {"curve": [], "curve_n": r["curve_n"],
                                        "base": [], "auroc": []})
                curves[sig]["curve"].append(r["curve"])
                curves[sig]["base"].append(r["base_accuracy"])
                curves[sig]["auroc"].append(r["auroc"])

            # per-class AUROC of the leading candidate, on the eval half
            _, ev = split_halves(correct.size, seed)
            s = scores["class_gap"]
            a_all = auroc(s[ev], correct[ev])
            direction = 1.0 if a_all >= 0.5 else -1.0
            for c in np.unique(q_lbl):
                m = ev[q_lbl[ev] == c]
                if m.size < 8 or correct[m].sum() in (0, m.size):
                    continue
                pc.setdefault(int(c), {"auroc": [], "n": [], "recall": []})
                pc[int(c)]["auroc"].append(auroc(direction * s[m], correct[m]))
                pc[int(c)]["n"].append(int(m.size))
                pc[int(c)]["recall"].append(float(correct[m].mean()))

        payload["curves"][cname] = {
            sig: {
                "coverage": CURVE.tolist(),
                "n_at_coverage": d["curve_n"],
                "accuracy_mean": np.mean(d["curve"], axis=0).tolist(),
                "accuracy_std": np.std(d["curve"], axis=0, ddof=1).tolist(),
                "base_accuracy": agg(d["base"]),
                "auroc": agg(d["auroc"]),
                "monotone_non_increasing_in_coverage": bool(
                    np.all(np.diff(np.mean(d["curve"], axis=0)) <= 1e-9)
                ),
            }
            for sig, d in curves.items()
        }
        payload["per_class"][cname] = {
            str(c): {"auroc": agg(v["auroc"]), "n_eval": int(np.mean(v["n"])),
                     "recall": agg(v["recall"])}
            for c, v in sorted(pc.items())
        }
        pcm = [v["auroc"]["mean"] for v in payload["per_class"][cname].values()
               if v["auroc"]["mean"] is not None]
        if pcm:
            print(f"  per-class class_gap AUROC: {len(pcm)} classes with enough support, "
                  f"median {np.median(pcm):.3f}, above 0.5 in {sum(x > 0.5 for x in pcm)}/{len(pcm)}")

        # ── failure corners, seed 42 ─────────────────────────────────────────
        b = built_by_seed[args.seeds[-1]]
        scores, pred = compute_all(b["dists"], b["gallery_labels"], b["stats"], k=best_k)
        correct = (pred == b["query_labels"]).astype(np.int64)
        paths = json.loads((args.features / "tgt_val_paths.json").read_text())
        corners = {}
        for sig in ("class_gap", "neg_d1"):
            s = scores[sig]
            hi = s >= np.nanquantile(s, 0.9)
            lo = s <= np.nanquantile(s, 0.1)
            corners[sig] = {
                "high_signal_wrong": {
                    "count": int((hi & (correct == 0)).sum()),
                    "share_of_high": float((correct[hi] == 0).mean()),
                    "examples": [Path(paths[i]).name for i in np.flatnonzero(hi & (correct == 0))[:5]],
                },
                "low_signal_correct": {
                    "count": int((lo & (correct == 1)).sum()),
                    "share_of_low": float((correct[lo] == 1).mean()),
                    "examples": [Path(paths[i]).name for i in np.flatnonzero(lo & (correct == 1))[:5]],
                },
                "accuracy_top_decile": float(correct[hi].mean()),
                "accuracy_bottom_decile": float(correct[lo].mean()),
                "accuracy_overall": float(correct.mean()),
            }
        payload["failures"][cname] = corners
        cg = corners["class_gap"]
        print(f"  class_gap deciles: top {cg['accuracy_top_decile']:.3f} / "
              f"overall {cg['accuracy_overall']:.3f} / bottom {cg['accuracy_bottom_decile']:.3f}")

    payload["meta"] = {
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": {"edge-reliability-gate": git_hash(PROJECT), "plantdiag-edge": git_hash(PLANTDIAG)},
        "k_grid": list(k_grid),
        "coverage_grid": CURVE.tolist(),
        "seeds": args.seeds,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    (args.out / "deep_dive.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwritten -> {args.out / 'deep_dive.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
