#!/usr/bin/env python3
"""S3 step 2: cross-dataset replication of the S2 causal result.

Queries are always the held-out 20 % of the 13 conservatively aligned PlantWild
classes — the same images in every cell, so the comparison across gallery
domains is paired within a seed.

Factors:
    S  gallery domain    plantwild (matched) | plantvillage (lab) | plantdoc (other field)
    G  gallery format    float32 | pq3bit
    Q  backbone          onnx_fp32 | onnx_int8
    A  adaptation        none (primary) | icoral (declared robustness arm)

Gallery size and per-class counts are equalised across all three domains at
`--per-class` images per class, so `S` is a domain contrast and not a size one.

Nothing here reads a correctness label before it is used as an outcome: the
gallery is built from images alone, k comes from a rule fixed in the
preregistration, and signal direction is chosen on a tuning half disjoint from
the half every reported number comes from.
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

from plantdiag_edge.domain_adapter.coral import IncrementalCORAL  # noqa: E402
from plantdiag_edge.polarquant import PolarQuant, rfwht  # noqa: E402
from reliability_gate.metrics import aurc, auroc, risk_coverage  # noqa: E402
from reliability_gate.signals import GalleryStats, compute_all  # noqa: E402
from s2_run_factorial import intra_class_nn, squared_l2, stratified_match  # noqa: E402

COVERAGES = np.array([0.10, 0.25, 0.50, 0.75, 1.00])
RFWHT_SEED = 42
SEEDS = (0, 1, 2, 3, 42)
PER_CLASS = 35  # binding minimum across the three domains, see S3_DATASET_INVENTORY
K_FIXED = 10  # Analysis A of the preregistration


def git_hash(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def auprc(scores: np.ndarray, correct: np.ndarray) -> float:
    """Average precision for `correct` ranked by `scores` (higher first)."""
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    y = np.asarray(correct, dtype=np.float64)[order]
    if y.sum() == 0:
        return float("nan")
    tp = np.cumsum(y)
    precision = tp / np.arange(1, y.size + 1)
    return float((precision * y).sum() / y.sum())


def load_pools(s2_dir: Path, s3_dir: Path) -> dict:
    """Feature pools of the three gallery domains plus the PlantWild queries."""
    s2 = {p.stem: np.load(p) for p in s2_dir.glob("*.npy")}
    s3 = {p.stem: np.load(p) for p in s3_dir.glob("*.npy")}
    tr_idx = s3["closed_train_idx"]
    q_idx = s3["closed_query_idx"]
    closed_lab = s3["closed_labels"]
    return {
        "s2": s2,
        "s3": s3,
        "query_idx": q_idx,
        "train_idx": tr_idx,
        "query_labels": closed_lab[q_idx],
        "pw_gallery_labels": closed_lab[tr_idx],
    }


def build_cell(cfg: dict, pools: dict, seed: int, per_class: int, k_rule: str) -> dict:
    fset = cfg["Q"]
    q_feat = pools["s3"][f"closed_{fset}"][pools["query_idx"]].astype(np.float32)
    q_lbl = pools["query_labels"]
    classes = np.unique(q_lbl)
    n_classes = int(max(classes)) + 1
    counts = {int(c): per_class for c in classes}

    if cfg["S"] == "plantwild":
        pool_f = pools["s3"][f"closed_{fset}"][pools["train_idx"]].astype(np.float32)
        pool_l = pools["pw_gallery_labels"]
    elif cfg["S"] == "plantvillage":
        pool_f = pools["s2"][f"src_{fset}"].astype(np.float32)
        pool_l = pools["s2"]["src_labels"]
    elif cfg["S"] == "plantdoc":
        pool_f = pools["s2"][f"tgt_tr_{fset}"].astype(np.float32)
        pool_l = pools["s2"]["tgt_tr_labels"]
    else:
        raise ValueError(cfg["S"])

    sel = stratified_match(pool_l, counts, seed)
    g_feat, g_lbl = pool_f[sel], pool_l[sel]
    got = {int(c): int((g_lbl == c).sum()) for c in classes}
    if min(got.values()) < per_class:
        raise RuntimeError(f"{cfg['S']} short of {per_class}/class: {got}")

    if cfg["A"] == "icoral":
        d = q_feat.shape[1]
        src = pools["s2"][f"src_{fset}"].astype(np.float32)
        source_cov = np.cov(src.T) + np.eye(d)
        ic = IncrementalCORAL(d)
        pw_tr = pools["s3"][f"closed_{fset}"][pools["train_idx"]].astype(np.float32)
        for i in range(0, len(pw_tr), 16):
            ic.update(pw_tr[i : i + 16])
        q_feat = ic.align_to_source(q_feat, source_cov).astype(np.float32)
        if cfg["S"] != "plantvillage":
            g_feat = ic.align_to_source(g_feat, source_cov).astype(np.float32)

    extra: dict = {}
    if cfg["G"] == "pq3bit":
        pq = PolarQuant(seed=RFWHT_SEED)
        codes, vmins, vmaxs = pq.encode(g_feat)
        g_eff = pq.decode(codes, vmins, vmaxs).astype(np.float32)
        q_eff = rfwht(q_feat, seed=RFWHT_SEED).astype(np.float32)
        extra["gallery_bytes"] = int(codes.nbytes + vmins.nbytes + vmaxs.nbytes)
    else:
        g_eff, q_eff = g_feat, q_feat
        extra["gallery_bytes"] = int(g_feat.nbytes)

    dists = squared_l2(q_eff, g_eff)
    stats = GalleryStats.from_intra_class_nn(intra_class_nn(g_eff, g_lbl), n_classes)
    k = K_FIXED if k_rule == "fixed" else max(1, round(len(g_lbl) / len(classes)))
    return {
        "dists": dists, "gallery_labels": g_lbl, "query_labels": q_lbl,
        "stats": stats, "n_classes": n_classes, "k": int(min(k, len(g_lbl))),
        "extra": extra, "gallery_effective": g_eff, "query_effective": q_eff,
    }


def evaluate_cell(built: dict, seed: int) -> dict:
    dists, g_lbl, q_lbl = built["dists"], built["gallery_labels"], built["query_labels"]
    scores, pred = compute_all(dists, g_lbl, built["stats"], k=built["k"])
    correct = (pred == q_lbl).astype(np.int64)
    classes = np.unique(q_lbl)

    per_class = {int(c): float((pred[q_lbl == c] == c).mean()) for c in classes}
    out = {
        "k": built["k"],
        "n_gallery": int(len(g_lbl)),
        "n_queries": int(len(q_lbl)),
        "n_classes": int(len(classes)),
        "accuracy": float(correct.mean()),
        "macro_recall": float(np.mean(list(per_class.values()))),
        "per_class_recall": per_class,
        "tie_share_d1_d2": float(
            (np.sort(dists, axis=1)[:, 0] == np.sort(dists, axis=1)[:, 1]).mean()
        ),
        **built["extra"],
        "signals": {},
        "signal_correlations": {},
    }

    from scipy.stats import spearmanr

    base = {n: scores[n] for n in ("neg_d1", "margin_rel") if n in scores}
    for name, s in scores.items():
        out["signal_correlations"][name] = {}
        for bn, b in base.items():
            m = np.isfinite(s) & np.isfinite(b)
            out["signal_correlations"][name][bn] = (
                float(spearmanr(s[m], b[m]).statistic) if m.sum() > 2 else None
            )

    rng = np.random.default_rng(seed)
    idx = rng.permutation(correct.size)
    cut = correct.size // 2
    tune_all, eval_all = idx[:cut], idx[cut:]
    for name, s in scores.items():
        finite = np.isfinite(s)
        tune, ev = tune_all[finite[tune_all]], eval_all[finite[eval_all]]
        if tune.size == 0 or ev.size == 0 or correct[ev].sum() in (0, ev.size):
            out["signals"][name] = {"auroc": None, "reason": "degenerate split"}
            continue
        a_tune = auroc(s[tune], correct[tune])
        direction = 1.0 if (np.isnan(a_tune) or a_tune >= 0.5) else -1.0
        se, ce = direction * s[ev], correct[ev]
        out["signals"][name] = {
            "auroc": auroc(se, ce),
            "auroc_undirected": auroc(s[ev], ce),
            "auprc": auprc(se, ce),
            "direction": direction,
            "aurc": aurc(se, ce),
            "aurc_random": aurc(np.zeros_like(se), ce),
            "accuracy_at_coverage": {
                f"{c:.2f}": float(a) for c, a in zip(COVERAGES, risk_coverage(se, ce, COVERAGES))
            },
            "n_at_coverage": {f"{c:.2f}": int(max(1, round(c * ce.size))) for c in COVERAGES},
            "n_eval": int(ev.size),
        }
    return out


def build_configs() -> list[dict]:
    cfgs = []
    for a in ("none", "icoral"):
        for s in ("plantwild", "plantvillage", "plantdoc"):
            for g in ("float32", "pq3bit"):
                for q in ("onnx_fp32", "onnx_int8"):
                    cfgs.append({
                        "name": f"s3_{a[:2]}_{s[:6]}_{g[:3]}_{q[-4:]}",
                        "A": a, "S": s, "G": g, "Q": q,
                    })
    return cfgs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s2-features", type=Path, default=PROJECT / "results" / "s2_features")
    ap.add_argument("--s3-features", type=Path, default=PROJECT / "results" / "s3_features")
    ap.add_argument("--out", type=Path, default=PROJECT / "results" / "s3_factorial")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--per-class", type=int, default=PER_CLASS)
    ap.add_argument("--k-rule", choices=["fixed", "structural"], default="fixed")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pools = load_pools(args.s2_features, args.s3_features)
    align = json.loads((PROJECT / "label_alignment.json").read_text())
    configs = build_configs()
    suffix = "" if args.k_rule == "fixed" else "_kstruct"

    print(f"queries {len(pools['query_labels'])}, classes {len(np.unique(pools['query_labels']))}, "
          f"gallery {args.per_class}/class, k-rule {args.k_rule}")
    t0, done = time.perf_counter(), 0
    for cfg in configs:
        for seed in args.seeds:
            dst = args.out / f"{cfg['name']}{suffix}_seed{seed}.json"
            if dst.exists():
                done += 1
                continue
            t = time.perf_counter()
            built = build_cell(cfg, pools, seed, args.per_class, args.k_rule)
            res = evaluate_cell(built, seed)
            res.update({"config": cfg["name"], "factors": {k: cfg[k] for k in ("A", "S", "G", "Q")},
                        "seed": seed, "k_rule": args.k_rule, "per_class": args.per_class})
            res["meta"] = {
                "command": " ".join(sys.argv),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "git": {"edge-reliability-gate": git_hash(PROJECT),
                        "plantdiag-edge": git_hash(PLANTDIAG)},
                "alignment": {"version": align["mapping_version"], "sha256": align["mapping_sha256"]},
                "s2_features": str(args.s2_features),
                "s3_features": str(args.s3_features),
                "wall_s": round(time.perf_counter() - t, 2),
                "environment": {"hostname": socket.gethostname(), "machine": platform.machine(),
                                "python": platform.python_version(), "numpy": np.__version__},
            }
            dst.write_text(json.dumps(res, indent=2, ensure_ascii=False))
            done += 1
            nd = res["signals"]["neg_d1"]["auroc_undirected"]
            cg = res["signals"]["class_gap"]["auroc"]
            print(f"[{done}/{len(configs) * len(args.seeds)}] {cfg['name']} seed{seed}  "
                  f"acc={res['accuracy']:.4f}  neg_d1(und)={nd:.3f}  class_gap={cg:.3f}", flush=True)
    print(f"\n{done} cell-runs in {time.perf_counter() - t0:.0f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
