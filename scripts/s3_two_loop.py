#!/usr/bin/env python3
"""S3 step 5: the two-loop hypothesis and the OOD tiers, on PlantWild alone.

Both OOD tiers come from PlantWild itself, so the contrast between "inside the
label space" and "outside" is not confounded with a change of dataset, camera or
collection protocol — which the S2 OOD check (FCDD fruit, beans) could not avoid.

  `known_crop_unknown_disease`  31 classes, crop in the label space, disease not
  `unseen_crop`                 69 classes, crop absent from the label space

Two questions, kept apart because they are different tasks:

  correctness  among in-space queries, does the signal rank correct above wrong?
  membership   does the signal separate in-space queries from out-of-space ones?

The preregistered complementarity criterion, fixed before these numbers existed:
|Spearman| between the two candidate signals < 0.5, and each must beat the other
by at least 0.10 AUROC on its own task.

The decision policy is the single pre-declared form — reject on membership, then
gate on reliability. Its thresholds are fitted on the tuning half and the
reported figure comes from the evaluation half, used once.
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
PER_CLASS = 35
K = 10
LOOP1 = "neg_d1"       # membership candidate: distance scale
LOOP2 = "class_gap"    # reliability candidate: class competition
CORR_MAX = 0.5         # preregistered
ADVANTAGE_MIN = 0.10   # preregistered


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
            "ci95": [lo, hi], "all_above_half": bool((a > 0.5).all()), "per_seed": a.tolist()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s2-features", type=Path, default=PROJECT / "results" / "s2_features")
    ap.add_argument("--s3-features", type=Path, default=PROJECT / "results" / "s3_features")
    ap.add_argument("--out", type=Path, default=PROJECT / "results" / "s3_two_loop")
    ap.add_argument("--fset", default="onnx_fp32")
    ap.add_argument("--gallery", choices=["plantwild", "plantvillage", "plantdoc"], default="plantwild")
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
    n_classes = int(max(classes)) + 1

    q_feat = s3[f"closed_{args.fset}"][s3["closed_query_idx"]].astype(np.float32)
    q_lbl = s3["closed_labels"][s3["closed_query_idx"]]
    ood = {t: s3[f"{t}_{args.fset}"].astype(np.float32)
           for t in ("ood_known_crop", "ood_unseen_crop")}

    if args.gallery == "plantwild":
        pool = (s3[f"closed_{args.fset}"][s3["closed_train_idx"]].astype(np.float32),
                s3["closed_labels"][s3["closed_train_idx"]])
    elif args.gallery == "plantvillage":
        pool = (s2[f"src_{args.fset}"].astype(np.float32), s2["src_labels"])
    else:
        pool = (s2[f"tgt_tr_{args.fset}"].astype(np.float32), s2["tgt_tr_labels"])

    print(f"gallery {args.gallery} @{PER_CLASS}/class, queries {len(q_lbl)}, "
          f"OOD known-crop {len(ood['ood_known_crop'])}, unseen-crop {len(ood['ood_unseen_crop'])}\n")

    corr_task, memb_task, spear, policy = {}, {}, [], []
    for seed in args.seeds:
        sel = stratified_match(pool[1], {int(c): PER_CLASS for c in classes}, seed)
        g_feat, g_lbl = pool[0][sel], pool[1][sel]
        pq = PolarQuant(seed=RFWHT_SEED)
        codes, vmins, vmaxs = pq.encode(g_feat)
        g_eff = pq.decode(codes, vmins, vmaxs).astype(np.float32)
        stats = GalleryStats.from_intra_class_nn(intra_class_nn(g_eff, g_lbl), n_classes)

        def score(feats):
            qe = rfwht(feats, seed=RFWHT_SEED).astype(np.float32)
            return compute_all(squared_l2(qe, g_eff), g_lbl, stats, k=K)

        in_scores, pred = score(q_feat)
        correct = (pred == q_lbl).astype(np.int64)
        ood_scores = {t: score(f)[0] for t, f in ood.items()}

        rng = np.random.default_rng(seed)
        idx = rng.permutation(correct.size)
        tune, ev = idx[: correct.size // 2], idx[correct.size // 2 :]

        for sig, s in in_scores.items():
            a_t = auroc(s[tune], correct[tune])
            d = 1.0 if (np.isnan(a_t) or a_t >= 0.5) else -1.0
            corr_task.setdefault(sig, []).append(auroc(d * s[ev], correct[ev]))
            for t, os_ in ood_scores.items():
                y = np.concatenate([np.ones(s.size, int), np.zeros(os_[sig].size, int)])
                memb_task.setdefault(t, {}).setdefault(sig, []).append(
                    auroc(np.concatenate([s, os_[sig]]), y))

        from scipy.stats import spearmanr
        m = np.isfinite(in_scores[LOOP1]) & np.isfinite(in_scores[LOOP2])
        spear.append(float(spearmanr(in_scores[LOOP1][m], in_scores[LOOP2][m]).statistic))

        # Pre-declared two-stage policy. Both thresholds fitted on `tune` only.
        memb = in_scores[LOOP1]
        rel = in_scores[LOOP2]
        a_t = auroc(rel[tune], correct[tune])
        d_rel = 1.0 if a_t >= 0.5 else -1.0
        pooled_ood = np.concatenate([ood_scores[t][LOOP1] for t in ood])
        # Stage 1 threshold: the membership quantile that keeps 90 % of tuning
        # in-space queries — chosen without ever looking at an OOD score.
        thr_memb = float(np.quantile(memb[tune], 0.10))
        # Stage 2 threshold: serve the most reliable half of what survives.
        surv = tune[memb[tune] >= thr_memb]
        thr_rel = float(np.quantile((d_rel * rel)[surv], 0.50)) if surv.size else -np.inf

        served = ev[(memb[ev] >= thr_memb) & ((d_rel * rel)[ev] >= thr_rel)]
        ood_admitted = float((pooled_ood >= thr_memb).mean())
        policy.append({
            "coverage_of_in_space": float(served.size / ev.size),
            "accuracy_on_served": float(correct[served].mean()) if served.size else float("nan"),
            "accuracy_baseline": float(correct[ev].mean()),
            "ood_admitted_share": ood_admitted,
            "n_served": int(served.size),
        })

    sigs = sorted(corr_task)
    payload = {
        "gallery": args.gallery, "fset": args.fset, "k": K, "per_class": PER_CLASS,
        "n_queries": int(len(q_lbl)),
        "n_ood": {t: int(len(f)) for t, f in ood.items()},
        "correctness_auroc": {s: agg(v) for s, v in corr_task.items()},
        "membership_auroc": {t: {s: agg(v) for s, v in d.items()} for t, d in memb_task.items()},
        "spearman_loop1_loop2": agg(spear),
        "complementarity_test": {
            "criterion": f"|spearman| < {CORR_MAX} and each signal beats the other by "
                         f">= {ADVANTAGE_MIN} AUROC on its own task",
            "loop1_signal": LOOP1, "loop2_signal": LOOP2,
        },
        "policy_pre_declared": {
            "form": "reject if membership below the 10th tuning percentile; then serve the "
                    "most reliable half of survivors",
            "thresholds_fitted_on": "tuning half only",
            "evaluated_on": "evaluation half, once",
            "per_seed": policy,
            "coverage_of_in_space": agg([p["coverage_of_in_space"] for p in policy]),
            "accuracy_on_served": agg([p["accuracy_on_served"] for p in policy]),
            "accuracy_baseline": agg([p["accuracy_baseline"] for p in policy]),
            "ood_admitted_share": agg([p["ood_admitted_share"] for p in policy]),
        },
        "meta": {
            "command": " ".join(sys.argv),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git": {"edge-reliability-gate": git_hash(PROJECT)},
            "alignment": {"version": align["mapping_version"], "sha256": align["mapping_sha256"]},
            "seeds": args.seeds, "wall_s": round(time.perf_counter() - t0, 1),
            "environment": {"python": platform.python_version(), "numpy": np.__version__},
        },
    }

    c1 = payload["correctness_auroc"]
    m1 = payload["membership_auroc"]
    adv_corr = c1[LOOP2]["mean"] - c1[LOOP1]["mean"]
    adv_memb = min(m1[t][LOOP1]["mean"] - m1[t][LOOP2]["mean"] for t in m1)
    sp = payload["spearman_loop1_loop2"]["mean"]
    payload["complementarity_test"].update({
        "spearman": sp,
        "loop2_advantage_on_correctness": adv_corr,
        "loop1_advantage_on_membership_min_over_tiers": adv_memb,
        "passes": bool(abs(sp) < CORR_MAX and adv_corr >= ADVANTAGE_MIN and adv_memb >= ADVANTAGE_MIN),
    })
    (args.out / f"two_loop_{args.gallery}_{args.fset}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"{'signal':<20}{'correctness':>13}{'known-crop':>13}{'unseen-crop':>13}")
    for s in sigs:
        print(f"{s:<20}{c1[s]['mean']:>13.3f}"
              f"{m1['ood_known_crop'][s]['mean']:>13.3f}{m1['ood_unseen_crop'][s]['mean']:>13.3f}")
    ct = payload["complementarity_test"]
    print(f"\ncomplementarity: spearman({LOOP1},{LOOP2}) = {sp:+.3f}")
    print(f"  {LOOP2} beats {LOOP1} on correctness by {adv_corr:+.3f}")
    print(f"  {LOOP1} beats {LOOP2} on membership by {adv_memb:+.3f} (worst tier)")
    print(f"  criterion passes: {ct['passes']}")
    p = payload["policy_pre_declared"]
    print(f"\npre-declared policy: coverage {p['coverage_of_in_space']['mean']:.3f}, "
          f"accuracy {p['accuracy_on_served']['mean']:.3f} vs baseline "
          f"{p['accuracy_baseline']['mean']:.3f}, OOD admitted {p['ood_admitted_share']['mean']:.3f}")
    print(f"\nwritten -> {args.out}  ({payload['meta']['wall_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
