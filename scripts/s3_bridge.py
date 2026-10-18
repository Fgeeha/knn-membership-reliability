#!/usr/bin/env python3
"""S3 step 4: why the S2 inversion did not reproduce — query domain or gallery size?

The preregistered S3 endpoint came back null: with PlantWild queries the built-in
confidence is not inverted for any gallery domain. Two candidate explanations
survive, and they are separable with the caches already on disk:

  **query domain** — the inversion is a property of the PlantVillage->PlantDoc
  pair and does not generalise;
  **gallery size** — S3 equalised galleries at 35 images per class, while the S2
  cells that inverted used 81 and 1418 per class. S2's own numbers trend that
  way (0.467 at the shallowest source gallery, 0.379 at the deepest), so size is
  a live confound rather than a post-hoc excuse.

This runs the 2x2 that separates them, on one class set (the 13 aligned classes),
one gallery size grid, and everything else held fixed:

    query domain   {plantdoc, plantwild}
    gallery domain {own (matched), plantvillage (mismatched)}
    per-class size {35, 60, 100, 200, 373}   where the pool allows

A matched gallery is capped by its own pool, so the size grid is reported per arm
with the sizes actually achieved rather than silently truncated.
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
from reliability_gate.metrics import auroc, bootstrap_ci  # noqa: E402
from reliability_gate.signals import GalleryStats, compute_all  # noqa: E402
from s2_run_factorial import intra_class_nn, squared_l2, stratified_match  # noqa: E402

RFWHT_SEED = 42
SEEDS = (0, 1, 2, 3, 42)
SIZE_GRID = (35, 60, 100, 200, 373)
K = 10


def git_hash(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def agg(v) -> dict:
    a = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if a.size == 0:
        return {"mean": None}
    lo, hi = bootstrap_ci(a) if a.size > 1 else (float(a[0]), float(a[0]))
    return {"mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "ci95": [lo, hi], "min": float(a.min()), "max": float(a.max()),
            "all_below_half": bool((a < 0.5).all()), "per_seed": a.tolist()}


def run(q_feat, q_lbl, pool_f, pool_l, classes, per_class, seed, compress: bool) -> dict:
    counts = {int(c): per_class for c in classes}
    sel = stratified_match(pool_l, counts, seed)
    g_feat, g_lbl = pool_f[sel], pool_l[sel]
    got = min(int((g_lbl == c).sum()) for c in classes)
    if compress:
        pq = PolarQuant(seed=RFWHT_SEED)
        codes, vmins, vmaxs = pq.encode(g_feat)
        g_eff = pq.decode(codes, vmins, vmaxs).astype(np.float32)
        q_eff = rfwht(q_feat, seed=RFWHT_SEED).astype(np.float32)
    else:
        g_eff, q_eff = g_feat, q_feat
    dists = squared_l2(q_eff, g_eff)
    n_classes = int(max(classes)) + 1
    stats = GalleryStats.from_intra_class_nn(intra_class_nn(g_eff, g_lbl), n_classes)
    scores, pred = compute_all(dists, g_lbl, stats, k=min(K, len(g_lbl)))
    correct = (pred == q_lbl).astype(np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(correct.size)
    ev = idx[correct.size // 2 :]
    return {
        "n_gallery": int(len(g_lbl)), "min_per_class": got,
        "accuracy": float(correct.mean()),
        "neg_d1_undirected": auroc(scores["neg_d1"][ev], correct[ev]),
        "class_gap_undirected": auroc(scores["class_gap"][ev], correct[ev]),
        "z_class_undirected": auroc(scores["z_class"][ev], correct[ev]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s2-features", type=Path, default=PROJECT / "results" / "s2_features")
    ap.add_argument("--s3-features", type=Path, default=PROJECT / "results" / "s3_features")
    ap.add_argument("--out", type=Path, default=PROJECT / "results" / "s3_bridge")
    ap.add_argument("--fset", default="onnx_fp32")
    ap.add_argument("--compress", action="store_true", help="use a PolarQuant gallery")
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

    # PlantDoc queries, restricted to the same 13 classes.
    pd_val_lbl = s2["tgt_val_labels"]
    pd_mask = np.isin(pd_val_lbl, classes)
    queries = {
        "plantdoc": (s2[f"tgt_val_{args.fset}"][pd_mask].astype(np.float32), pd_val_lbl[pd_mask]),
        "plantwild": (s3[f"closed_{args.fset}"][s3["closed_query_idx"]].astype(np.float32),
                      s3["closed_labels"][s3["closed_query_idx"]]),
    }
    own_pool = {
        "plantdoc": (s2[f"tgt_tr_{args.fset}"].astype(np.float32), s2["tgt_tr_labels"]),
        "plantwild": (s3[f"closed_{args.fset}"][s3["closed_train_idx"]].astype(np.float32),
                      s3["closed_labels"][s3["closed_train_idx"]]),
    }
    pv_pool = (s2[f"src_{args.fset}"].astype(np.float32), s2["src_labels"])

    print(f"classes {len(classes)}  queries: plantdoc {pd_mask.sum()}, "
          f"plantwild {len(queries['plantwild'][1])}  features {args.fset}  "
          f"gallery {'PolarQuant-3bit' if args.compress else 'float32'}\n")

    rows = []
    for qname, (qf, ql) in queries.items():
        for gname in ("own", "plantvillage"):
            pool = own_pool[qname] if gname == "own" else pv_pool
            avail = min(int((pool[1] == c).sum()) for c in classes)
            for size in SIZE_GRID:
                if size > avail:
                    continue
                per_seed = [run(qf, ql, pool[0], pool[1], classes, size, s, args.compress)
                            for s in args.seeds]
                rows.append({
                    "query_domain": qname,
                    "gallery_domain": "own(matched)" if gname == "own" else "plantvillage(mismatched)",
                    "per_class": size,
                    "pool_min_per_class": avail,
                    "n_gallery": per_seed[0]["n_gallery"],
                    "accuracy": agg([r["accuracy"] for r in per_seed]),
                    "neg_d1_undirected": agg([r["neg_d1_undirected"] for r in per_seed]),
                    "class_gap_undirected": agg([r["class_gap_undirected"] for r in per_seed]),
                    "z_class_undirected": agg([r["z_class_undirected"] for r in per_seed]),
                })

    payload = {
        "question": "does the S2 inversion depend on query domain or on gallery size?",
        "fset": args.fset, "compressed": bool(args.compress), "k": K,
        "n_classes": int(len(classes)),
        "n_queries": {k: int(len(v[1])) for k, v in queries.items()},
        "size_grid": list(SIZE_GRID),
        "rows": rows,
        "meta": {
            "command": " ".join(sys.argv),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git": {"edge-reliability-gate": git_hash(PROJECT)},
            "alignment": {"version": align["mapping_version"], "sha256": align["mapping_sha256"]},
            "seeds": args.seeds,
            "wall_s": round(time.perf_counter() - t0, 1),
            "environment": {"python": platform.python_version(), "numpy": np.__version__},
        },
    }
    tag = f"{args.fset}_{'pq' if args.compress else 'f32'}"
    (args.out / f"bridge_{tag}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"{'queries':<11}{'gallery':<26}{'N/cls':>6}{'N':>7}{'acc':>8}"
          f"{'neg_d1(und)':>14}{'inverted':>10}{'class_gap':>11}")
    for r in rows:
        nd = r["neg_d1_undirected"]
        print(f"{r['query_domain']:<11}{r['gallery_domain']:<26}{r['per_class']:>6}"
              f"{r['n_gallery']:>7}{r['accuracy']['mean']:>8.4f}"
              f"{nd['mean']:>9.3f}±{nd['std']:.3f}"
              f"{str(nd['mean'] < 0.5):>10}{r['class_gap_undirected']['mean']:>11.3f}")
    print(f"\nwritten -> {args.out / f'bridge_{tag}.json'}  ({payload['meta']['wall_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
