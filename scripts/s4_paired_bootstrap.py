#!/usr/bin/env python3
"""S4-D: per-query paired bootstrap for the key bridge cells (preregistered H8).

The spread published by S2/S3 is between-seed spread of gallery resampling at a
fixed query set; it does not capture the sampling uncertainty of the queries
themselves — the defect paper4 documented as its R7 result. This computes, for
the preregistered cells

    {plantdoc, plantwild} x plantvillage(mismatched) gallery x per-class {200, 373},

a 95 % percentile bootstrap CI of the undirected `neg_d1` AUROC over QUERIES
(10 000 resamples of the evaluation half), one CI per seed. H8 criterion, fixed
before running: the CI lies entirely below 0.5 for PlantDoc cells and entirely
above 0.5 for PlantWild cells on >= 4 of 5 seeds per cell.

The cell construction is byte-identical to `s3_bridge.py` (same stratified
gallery, same eval-half split, same k), only the aggregation differs.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
PLANTDIAG = PROJECT.parent / "plantdiag-edge"
sys.path.insert(0, str(PLANTDIAG))
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from plantdiag_edge.polarquant import PolarQuant, rfwht  # noqa: E402
from reliability_gate.metrics import auroc  # noqa: E402
from reliability_gate.signals import GalleryStats, compute_all  # noqa: E402
from s2_run_factorial import intra_class_nn, squared_l2, stratified_match  # noqa: E402

RFWHT_SEED = 42
SEEDS = (0, 1, 2, 3, 42)
SIZES = (200, 373)
K = 10
N_BOOT = 10_000
SIGNAL = "neg_d1"


def git_hash(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def cell_per_query(q_feat, q_lbl, pool_f, pool_l, classes, per_class, seed, compress):
    sel = stratified_match(pool_l, {int(c): per_class for c in classes}, seed)
    g_feat, g_lbl = pool_f[sel], pool_l[sel]
    if compress:
        pq = PolarQuant(seed=RFWHT_SEED)
        codes, vmins, vmaxs = pq.encode(g_feat)
        g_eff = pq.decode(codes, vmins, vmaxs).astype(np.float32)
        q_eff = rfwht(q_feat, seed=RFWHT_SEED).astype(np.float32)
    else:
        g_eff, q_eff = g_feat, q_feat
    stats = GalleryStats.from_intra_class_nn(intra_class_nn(g_eff, g_lbl), int(max(classes)) + 1)
    scores, pred = compute_all(squared_l2(q_eff, g_eff), g_lbl, stats, k=min(K, len(g_lbl)))
    correct = (pred == q_lbl).astype(np.int64)
    rng = np.random.default_rng(seed)
    ev = rng.permutation(correct.size)[correct.size // 2 :]
    return scores[SIGNAL][ev], correct[ev]


def boot_ci(s, y, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        vals[i] = auroc(s[idx], y[idx])
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s2-features", type=Path, default=PROJECT / "results" / "s2_features")
    ap.add_argument("--s3-features", type=Path, default=PROJECT / "results" / "s3_features")
    ap.add_argument("--out", type=Path, default=PROJECT / "results" / "s4_paired_bootstrap")
    ap.add_argument("--fset", default="onnx_fp32")
    ap.add_argument("--backbone-tag", default="c6", help="label written into the output")
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    s2 = {p.stem: np.load(p) for p in args.s2_features.glob("*.npy")}
    s3 = {p.stem: np.load(p) for p in args.s3_features.glob("*.npy")}
    align = json.loads((PROJECT / "label_alignment.json").read_text())
    spec = json.loads((PLANTDIAG / "configs" / "labels_clean.json").read_text())
    l2i = {l: i for i, l in enumerate(spec["target_labels"])}
    classes = np.array(sorted(l2i[v] for v in align["conservative_mapping"].values()))

    pd_lbl = s2["tgt_val_labels"]
    pd_mask = np.isin(pd_lbl, classes)
    queries = {
        "plantdoc": (s2[f"tgt_val_{args.fset}"][pd_mask].astype(np.float32), pd_lbl[pd_mask]),
        "plantwild": (s3[f"closed_{args.fset}"][s3["closed_query_idx"]].astype(np.float32),
                      s3["closed_labels"][s3["closed_query_idx"]]),
    }
    pv_pool = (s2[f"src_{args.fset}"].astype(np.float32), s2["src_labels"])
    expect_below = {"plantdoc": True, "plantwild": False}

    rows = []
    for qname, (qf, ql) in queries.items():
        for size in SIZES:
            per_seed = []
            for seed in args.seeds:
                s, y = cell_per_query(qf, ql, pv_pool[0], pv_pool[1], classes, size, seed, args.compress)
                a = auroc(s, y)
                lo, hi = boot_ci(s, y, seed)
                ok = (hi < 0.5) if expect_below[qname] else (lo > 0.5)
                per_seed.append({"seed": seed, "n_eval": int(len(y)), "auroc": a,
                                 "ci95": [lo, hi], "ci_on_expected_side": bool(ok)})
            n_ok = sum(1 for r in per_seed if r["ci_on_expected_side"])
            rows.append({
                "query_domain": qname, "gallery": "plantvillage(mismatched)",
                "per_class": size,
                "expected_side": "below 0.5" if expect_below[qname] else "above 0.5",
                "seeds_ci_on_expected_side": f"{n_ok}/{len(per_seed)}",
                "h8_cell": n_ok >= 4,
                "per_seed": per_seed,
            })
            print(f"{qname:>10} @{size}: " + "  ".join(
                f"{r['auroc']:.3f}[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]" for r in per_seed)
                + f"  -> {n_ok}/{len(per_seed)} on expected side")

    payload = {
        "hypothesis": "H8 (S4_PREREGISTRATION.md)",
        "signal": SIGNAL, "fset": args.fset, "backbone": args.backbone_tag,
        "compressed_gallery": bool(args.compress), "k": K, "n_boot": N_BOOT,
        "rows": rows,
        "h8_supported": bool(all(r["h8_cell"] for r in rows)),
        "meta": {
            "command": " ".join(sys.argv),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git": {"edge-reliability-gate": git_hash(PROJECT)},
            "seeds": args.seeds, "wall_s": round(time.perf_counter() - t0, 1),
            "environment": {"python": platform.python_version(), "numpy": np.__version__},
        },
    }
    tag = f"{args.backbone_tag}_{args.fset}_{'pq' if args.compress else 'f32'}"
    out = args.out / f"paired_bootstrap_{tag}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nH8 supported ({args.backbone_tag}): {payload['h8_supported']}")
    print(f"written -> {out}  ({payload['meta']['wall_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
