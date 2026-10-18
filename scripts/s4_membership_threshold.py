#!/usr/bin/env python3
"""S4-C: operating point of the membership circuit (preregistered H7).

S3 established that `neg_d1` separates in-space inputs from far OOD (fruit
tiers, AUROC 0.85-0.88) but barely sees near OOD (bean leaves, 0.58-0.66).
AUROC alone does not answer the deployment question: *is there a single
threshold that keeps almost all in-space inputs while refusing at least half
of the far OOD?* H7 (S4_PREREGISTRATION.md §2) fixes the answer criterion
before this script was run:

    threshold = a quantile of `neg_d1` fitted on the in-space TUNING half only;
    SUPPORTED if some threshold gives in-space acceptance >= 0.90 AND far-OOD
    rejection >= 0.50 on both fruit tiers, on >= 4 of 5 seeds;
    preregistered expectation-limitation: bean-leaf rejection < 0.20 there.

The preregistration did not fix the gallery domain, so both existing
configurations are evaluated (S=target `s2_1011`, S=source `s2_1010`) and H7 is
read conservatively: SUPPORTED only if the criterion holds in both.

Everything is recomputed from the cached features through the byte-identical
S2 cell-building path (`build_cell_arrays`), exactly as `s3_ood_check.py` did.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
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

from reliability_gate.signals import compute_all  # noqa: E402
from s2_run_factorial import build_cell_arrays, build_configs  # noqa: E402
from s3_ood_check import ood_distances  # noqa: E402

SEEDS = (0, 1, 2, 3, 42)
K = 10
QUANTILES = tuple(round(q, 3) for q in np.arange(0.01, 0.51, 0.01))
TIERS = ("beans", "fcdd_apple", "fcdd_other")
FAR = ("fcdd_apple", "fcdd_other")
CRIT_ACCEPT = 0.90
CRIT_REJECT = 0.50
CRIT_NEAR = 0.20


def git_hash(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=PROJECT / "results" / "s2_features")
    ap.add_argument("--ood-dir", type=Path, default=PROJECT / "results" / "s3_ood")
    ap.add_argument("--out", type=Path, default=PROJECT / "results" / "s4_membership")
    ap.add_argument("--signal", default="neg_d1")
    ap.add_argument("--configs", nargs="+", default=["s2_1011", "s2_1010"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    cache = {p.stem: np.load(p) for p in args.features.glob("*.npy")}
    ood_feat = {t: np.load(args.ood_dir / f"{t}_onnx_fp32.npy") for t in TIERS}
    cfgs = {c["name"]: c for c in build_configs()}

    report_cfg = {}
    for cname in args.configs:
        cfg = cfgs[cname]
        per_seed_rows = []
        for seed in args.seeds:
            built = build_cell_arrays(cfg, cache, seed)
            in_scores, _ = compute_all(built["dists"], built["gallery_labels"], built["stats"], k=K)
            s_in = in_scores[args.signal]
            ood_scores = {}
            for tier, feats in ood_feat.items():
                d_ood = ood_distances(cfg, cache, seed, feats)
                o, _ = compute_all(d_ood, built["gallery_labels"], built["stats"], k=K)
                ood_scores[tier] = o[args.signal]

            # tuning/evaluation halves of the in-space set: same split rule as
            # the bridge and two-loop (permutation by the cell seed)
            rng = np.random.default_rng(seed)
            idx = rng.permutation(s_in.size)
            tune, ev = idx[: s_in.size // 2], idx[s_in.size // 2 :]

            ops = []
            for q in QUANTILES:
                thr = float(np.quantile(s_in[tune], q))
                accept = float((s_in[ev] >= thr).mean())
                rej = {t: float((ood_scores[t] < thr).mean()) for t in TIERS}
                ops.append({"q": q, "threshold": thr, "in_space_acceptance": accept,
                            "rejection": rej,
                            "meets_h7": accept >= CRIT_ACCEPT and all(rej[t] >= CRIT_REJECT for t in FAR)})
            feasible = [o for o in ops if o["meets_h7"]]
            best = max(feasible, key=lambda o: min(o["rejection"][t] for t in FAR)) if feasible else None
            per_seed_rows.append({
                "seed": seed,
                "n_tune": int(tune.size), "n_eval": int(ev.size),
                "n_ood": {t: int(len(ood_scores[t])) for t in TIERS},
                "operating_curve": ops,
                "h7_feasible": bool(feasible),
                "best_operating_point": best,
            })

        n_ok = sum(1 for r in per_seed_rows if r["h7_feasible"])
        bests = [r["best_operating_point"] for r in per_seed_rows if r["best_operating_point"]]
        near = [b["rejection"]["beans"] for b in bests]
        report_cfg[cname] = {
            "factors": {kk: cfg[kk] for kk in ("A", "Q", "G", "S")},
            "seeds_with_feasible_point": f"{n_ok}/{len(per_seed_rows)}",
            "h7_main": n_ok >= 4,
            "near_ood_rejection_at_best": {
                "per_seed": near,
                "mean": float(np.mean(near)) if near else None,
                "below_020_all": bool(all(x < CRIT_NEAR for x in near)) if near else None,
            },
            "best_points_per_seed": bests,
            "per_seed": per_seed_rows,
        }
        print(f"{cname} (S={cfg['S']}): feasible on {n_ok}/{len(per_seed_rows)} seeds; "
              f"near-OOD rejection at best: {[round(x, 3) for x in near]}")

    h7_supported = all(v["h7_main"] for v in report_cfg.values())
    payload = {
        "hypothesis": "H7 (S4_PREREGISTRATION.md)",
        "signal": args.signal,
        "criteria": {"in_space_acceptance": CRIT_ACCEPT, "far_ood_rejection": CRIT_REJECT,
                     "near_ood_expected_below": CRIT_NEAR,
                     "reading": "conservative: both gallery domains must pass"},
        "quantile_grid": list(QUANTILES),
        "k": K,
        "configs": report_cfg,
        "h7_supported": bool(h7_supported),
        "meta": {
            "command": " ".join(sys.argv),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git": {"edge-reliability-gate": git_hash(PROJECT)},
            "seeds": args.seeds,
            "wall_s": round(time.perf_counter() - t0, 1),
            "environment": {"hostname": socket.gethostname(), "machine": platform.machine(),
                            "python": platform.python_version(), "numpy": np.__version__},
        },
    }
    out = args.out / f"membership_threshold_{args.signal}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nH7 supported (both configs): {h7_supported}")
    print(f"written -> {out}  ({payload['meta']['wall_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
