#!/usr/bin/env python3
"""S3 step 6: per-class evidence and failure structure, with an honest N floor.

S2's per-class section had 8-25 evaluation queries per class, which cannot
support a per-class AUROC: at n = 10 the standard error is around 0.15-0.19, so
values from 0.3 to 0.9 are one sampling fluctuation apart. The task explicitly
asks not to repeat that.

So the floor is declared here **before** the numbers are read:

    MIN_N_PER_CLASS = 25 evaluation queries.

Classes at or above it get an AUROC with a bootstrap interval. Classes below it
get counts and recall only, and are named as insufficient. No significance claim
is made for either group.

Direction of each signal is still chosen on the tuning half, but the per-class
figures are computed over the **full** query set, because halving it would put
every class under the floor. That is a deliberate trade: the direction cannot
leak (it is one bit, fitted on a disjoint half), while the per-class estimate
gains the sample size it needs.

Failure groups follow the four corners the task specifies, and each is described
by what the data can support: crop, disease, predicted class, whether the nearest
gallery entry belongs to the true class, and the distance profile.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections import Counter
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

MIN_N_PER_CLASS = 25  # declared before reading any per-class number
RFWHT_SEED = 42
SEEDS = (0, 1, 2, 3, 42)
PER_CLASS = 35
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
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "ci95": [lo, hi], "per_seed": a.tolist()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s2-features", type=Path, default=PROJECT / "results" / "s2_features")
    ap.add_argument("--s3-features", type=Path, default=PROJECT / "results" / "s3_features")
    ap.add_argument("--out", type=Path, default=PROJECT / "results" / "s3_per_class")
    ap.add_argument("--fset", default="onnx_fp32")
    ap.add_argument("--gallery", choices=["plantwild", "plantvillage"], default="plantwild")
    ap.add_argument("--signal", default="class_gap")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    s2 = {p.stem: np.load(p) for p in args.s2_features.glob("*.npy")}
    s3 = {p.stem: np.load(p) for p in args.s3_features.glob("*.npy")}
    align = json.loads((PROJECT / "label_alignment.json").read_text())
    spec = json.loads((PLANTDIAG / "configs" / "labels_clean.json").read_text())
    canonical = spec["target_labels"]
    l2i = {l: i for i, l in enumerate(canonical)}
    classes = np.array(sorted(l2i[v] for v in align["conservative_mapping"].values()))
    n_classes = len(canonical)

    qi = s3["closed_query_idx"]
    q_feat = s3[f"closed_{args.fset}"][qi].astype(np.float32)
    q_lbl = s3["closed_labels"][qi]
    src_cls = np.array(json.loads((args.s3_features / "closed_srcclass.json").read_text()))[qi]
    pool = ((s3[f"closed_{args.fset}"][s3["closed_train_idx"]].astype(np.float32),
             s3["closed_labels"][s3["closed_train_idx"]])
            if args.gallery == "plantwild"
            else (s2[f"src_{args.fset}"].astype(np.float32), s2["src_labels"]))

    per_class: dict[int, dict] = {}
    corners_all: list[dict] = []
    for seed in args.seeds:
        sel = stratified_match(pool[1], {int(c): PER_CLASS for c in classes}, seed)
        g_feat, g_lbl = pool[0][sel], pool[1][sel]
        pq = PolarQuant(seed=RFWHT_SEED)
        codes, vmins, vmaxs = pq.encode(g_feat)
        g_eff = pq.decode(codes, vmins, vmaxs).astype(np.float32)
        stats = GalleryStats.from_intra_class_nn(intra_class_nn(g_eff, g_lbl), n_classes)
        dists = squared_l2(rfwht(q_feat, seed=RFWHT_SEED).astype(np.float32), g_eff)
        scores, pred = compute_all(dists, g_lbl, stats, k=K)
        correct = (pred == q_lbl).astype(np.int64)

        rng = np.random.default_rng(seed)
        idx = rng.permutation(correct.size)
        tune = idx[: correct.size // 2]
        s = scores[args.signal]
        a_t = auroc(s[tune], correct[tune])
        d = 1.0 if (np.isnan(a_t) or a_t >= 0.5) else -1.0
        sd = d * s

        for c in classes:
            m = q_lbl == c
            e = per_class.setdefault(int(c), {"n": int(m.sum()), "recall": [], "auroc": []})
            e["recall"].append(float(correct[m].mean()))
            if m.sum() >= MIN_N_PER_CLASS and 0 < correct[m].sum() < m.sum():
                e["auroc"].append(auroc(sd[m], correct[m]))

        if seed == args.seeds[-1]:
            hi = sd >= np.nanquantile(sd, 0.9)
            lo = sd <= np.nanquantile(sd, 0.1)
            memb = scores["neg_d1"]
            mhi = memb >= np.nanquantile(memb, 0.9)
            mlo = memb <= np.nanquantile(memb, 0.1)
            nearest_same = g_lbl[np.argmin(dists, axis=1)] == q_lbl
            for gname, mask in (
                ("high_membership_wrong", mhi & (correct == 0)),
                ("low_membership_correct", mlo & (correct == 1)),
                ("high_reliability_wrong", hi & (correct == 0)),
                ("low_reliability_correct", lo & (correct == 1)),
            ):
                sub = np.flatnonzero(mask)
                corners_all.append({
                    "group": gname,
                    "n": int(sub.size),
                    "crops": dict(Counter(src_cls[i].split()[0] for i in sub).most_common(5)),
                    "true_classes": dict(Counter(canonical[q_lbl[i]] for i in sub).most_common(5)),
                    "predicted_classes": dict(Counter(canonical[pred[i]] for i in sub).most_common(5)),
                    "nearest_is_true_class_share": float(nearest_same[sub].mean()) if sub.size else None,
                    "d1_median": float(np.median(np.sort(dists[sub], axis=1)[:, 0])) if sub.size else None,
                    "d1_median_all": float(np.median(np.sort(dists, axis=1)[:, 0])),
                })

    ok = {c: v for c, v in per_class.items() if v["n"] >= MIN_N_PER_CLASS}
    thin = {c: v for c, v in per_class.items() if v["n"] < MIN_N_PER_CLASS}
    payload = {
        "gallery": args.gallery, "signal": args.signal, "fset": args.fset,
        "min_n_per_class_declared": MIN_N_PER_CLASS,
        "n_queries": int(len(q_lbl)),
        "classes_meeting_floor": len(ok),
        "classes_below_floor": len(thin),
        "note": "per-class figures use the full query set; only the signal direction "
                "comes from the tuning half, so the estimate is not halved",
        "per_class": {
            canonical[c]: {
                "n_queries": v["n"],
                "meets_floor": v["n"] >= MIN_N_PER_CLASS,
                "recall": agg(v["recall"]),
                "auroc": agg(v["auroc"]) if v["auroc"] else None,
            }
            for c, v in sorted(per_class.items())
        },
        "failure_groups": corners_all,
        "meta": {
            "command": " ".join(sys.argv),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git": {"edge-reliability-gate": git_hash(PROJECT)},
            "alignment": {"version": align["mapping_version"], "sha256": align["mapping_sha256"]},
            "seeds": args.seeds, "wall_s": round(time.perf_counter() - t0, 1),
            "environment": {"python": platform.python_version(), "numpy": np.__version__},
        },
    }
    (args.out / f"per_class_{args.gallery}_{args.signal}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"gallery {args.gallery}, signal {args.signal}, {len(q_lbl)} queries, "
          f"floor {MIN_N_PER_CLASS}/class")
    print(f"  {len(ok)} classes meet the floor, {len(thin)} below it\n")
    print(f"{'class':<32}{'N':>5}{'recall':>9}{'AUROC':>18}")
    for c, v in sorted(per_class.items(), key=lambda kv: -kv[1]["n"]):
        a = agg(v["auroc"]) if v["auroc"] else None
        astr = f"{a['mean']:.3f} ± {a['std']:.3f}" if a and a["mean"] is not None else "insufficient N"
        print(f"{canonical[c]:<32}{v['n']:>5}{np.mean(v['recall']):>9.3f}{astr:>18}")
    if ok:
        ms = [agg(v["auroc"])["mean"] for v in ok.values() if v["auroc"]]
        print(f"\nover classes meeting the floor: median {np.median(ms):.3f}, "
              f"above 0.5 in {sum(x > 0.5 for x in ms)}/{len(ms)}")
    print("\nfailure groups (last seed):")
    for g in corners_all:
        print(f"  {g['group']:<26} n={g['n']:<4} nearest-is-true={g['nearest_is_true_class_share']}"
              f"  d1_median={g['d1_median']:.1f} (all {g['d1_median_all']:.1f})")
        print(f"      top true classes: {list(g['true_classes'])[:3]}")
    print(f"\nwritten -> {args.out}  ({payload['meta']['wall_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
