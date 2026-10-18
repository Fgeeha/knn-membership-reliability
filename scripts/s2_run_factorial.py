#!/usr/bin/env python3
"""S2 step 2: run the 2^4 factorial and its controls over cached features.

Every cell shares the same 464-image evaluation set, the same 23-class label
space, the same preprocessing and the same correctness definition (top-1 equals
the true class). Only the four declared factors move:

    A  domain adaptation       none | icoral
    Q  backbone precision      onnx_fp32 | onnx_int8   (a real quantizer)
    G  gallery representation  float32 | pq3bit
    S  gallery domain          source (PlantVillage) | target (PlantDoc train)

The kNN head is held fixed in all sixteen cells, which is what lets `G` be read
as compression alone: the published A->C transition cannot separate compression
from the head change, because a linear head has no gallery at all.

Gallery size and class distribution are matched across the `S` arms — the source
arm is subsampled to the target arm's exact per-class counts — so `S` is a domain
contrast and not a size contrast. Controls with the full 32 605-vector source
gallery are run separately to connect to the published numbers.

Writes one JSON per (config, seed) plus a combined summary. No number is typed
by hand anywhere downstream.
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

from plantdiag_edge.domain_adapter.coral import IncrementalCORAL  # noqa: E402
from plantdiag_edge.polarquant import PolarQuant, rfwht  # noqa: E402
from reliability_gate.metrics import aurc, auroc, bootstrap_ci, risk_coverage  # noqa: E402
from reliability_gate.signals import GalleryStats, compute_all  # noqa: E402

COVERAGES = np.array([0.10, 0.25, 0.50, 0.75, 1.00])
RFWHT_SEED = 42  # must match the deployed bundle
SPLIT_SEEDS = (0, 1, 2, 3, 42)
K_DEFAULT = 10


def git_hash(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ── Distance computation ──────────────────────────────────────────────────────


def squared_l2(queries: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """(Q, N) squared euclidean distances via the gram identity, in float64.

    The reference `asymmetric_l2` in `plantdiag-edge` materialises the whole
    dequantized gallery for every single query, which is 267 MB per call at
    N = 32 605. The identity form is mathematically the same quantity; float64
    is used so the expansion does not lose the precision the subtraction form
    keeps. `verify_against_reference` checks the two agree.
    """
    q = np.asarray(queries, dtype=np.float64)
    g = np.asarray(gallery, dtype=np.float64)
    d = (q * q).sum(1)[:, None] - 2.0 * (q @ g.T) + (g * g).sum(1)[None, :]
    return np.maximum(d, 0.0)


def verify_against_reference(
    queries: np.ndarray, codes: np.ndarray, vmins: np.ndarray, vmaxs: np.ndarray,
    dequantized: np.ndarray, padded_dim: int, n_check: int = 8,
) -> dict:
    """Confirm the fast path reproduces `plantdiag-edge`'s asymmetric distance.

    `padded_dim` is mandatory: `PolarQuant.encode` bit-packs by default, so the
    codes are ceil(p*3/8) bytes wide and the reference cannot recover `p` from
    their shape. Omitting it does not silently mis-decode — it raises a broadcast
    error — but the argument is required here so it cannot be forgotten.
    """
    from plantdiag_edge.polarquant.decoder import asymmetric_l2

    idx = np.linspace(0, len(queries) - 1, min(n_check, len(queries))).astype(int)
    max_rel, rank_ok = 0.0, True
    for i in idx:
        ref = asymmetric_l2(queries[i].astype(np.float32), codes, vmins, vmaxs, padded_dim)
        fast = squared_l2(queries[i : i + 1], dequantized)[0]
        denom = np.maximum(np.abs(ref), 1e-6)
        max_rel = max(max_rel, float((np.abs(fast - ref) / denom).max()))
        rank_ok &= bool(np.array_equal(np.argsort(ref, kind="mergesort")[:20],
                                       np.argsort(fast, kind="mergesort")[:20]))
    return {"n_checked": int(idx.size), "max_relative_deviation": max_rel, "top20_order_identical": rank_ok}


# ── Gallery construction ──────────────────────────────────────────────────────


def stratified_match(
    labels: np.ndarray, target_counts: dict[int, int], seed: int
) -> np.ndarray:
    """Indices reproducing `target_counts` per class, sampled without replacement."""
    rng = np.random.default_rng(seed)
    out = []
    for c, want in sorted(target_counts.items()):
        pool = np.flatnonzero(labels == c)
        if pool.size == 0:
            continue
        take = min(want, pool.size)
        out.append(rng.choice(pool, size=take, replace=False))
    return np.sort(np.concatenate(out)) if out else np.array([], dtype=int)


def intra_class_nn(gallery: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    """Nearest same-class neighbour distance for every gallery entry, class by class."""
    out = {}
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        if idx.size < 2:
            continue
        d = squared_l2(gallery[idx], gallery[idx])
        np.fill_diagonal(d, np.inf)
        out[int(c)] = d.min(axis=1)
    return out


# ── One cell ──────────────────────────────────────────────────────────────────


def build_cell_arrays(cfg: dict, cache: dict, seed: int, verify: bool = False) -> dict:
    """Construct one factorial cell: gallery, adapted query, distances, statistics.

    Split out of `run_cell` so the deep-dive analysis can reach the per-sample
    arrays without duplicating the construction. `run_cell` calls it, so any
    divergence would show up as a mismatch in `verify_s2_results.py`, which
    recomputes stored cells through this same path.
    """
    fset = cfg["Q"]
    q_feat = cache[f"tgt_val_{fset}"].astype(np.float32)
    q_lbl = cache["tgt_val_labels"]
    tr_feat = cache[f"tgt_tr_{fset}"].astype(np.float32)
    src_feat = cache[f"src_{fset}"].astype(np.float32)
    src_lbl = cache["src_labels"]
    tr_lbl = cache["tgt_tr_labels"]
    n_classes = int(max(q_lbl.max(), src_lbl.max())) + 1

    # ── gallery selection (factor S, with matched size and class balance) ────
    if cfg["S"] == "target":
        g_feat_raw, g_lbl = tr_feat, tr_lbl
        g_is_target = True
    else:
        g_is_target = False
        if cfg.get("full_source"):
            g_feat_raw, g_lbl = src_feat, src_lbl
        else:
            counts = {int(c): int((tr_lbl == c).sum()) for c in np.unique(tr_lbl)}
            sel = stratified_match(src_lbl, counts, seed)
            g_feat_raw, g_lbl = src_feat[sel], src_lbl[sel]

    # ── domain adaptation (factor A) ─────────────────────────────────────────
    # Everything that lives in the target domain is mapped into source space, so
    # queries and gallery always end up in one space. A source gallery is already
    # there and is left untouched.
    if cfg["A"] == "icoral":
        d = q_feat.shape[1]
        source_cov = np.cov(src_feat.T) + np.eye(d)
        icoral = IncrementalCORAL(d)
        for i in range(0, len(tr_feat), 16):
            icoral.update(tr_feat[i : i + 16])
        q_feat = icoral.align_to_source(q_feat, source_cov).astype(np.float32)
        if g_is_target:
            g_feat = icoral.align_to_source(g_feat_raw, source_cov).astype(np.float32)
        else:
            g_feat = g_feat_raw
    else:
        g_feat = g_feat_raw

    # ── gallery representation (factor G) ────────────────────────────────────
    extra: dict = {}
    if cfg["G"] == "pq3bit":
        pq = PolarQuant(seed=RFWHT_SEED)
        codes, vmins, vmaxs = pq.encode(g_feat)
        g_eff = pq.decode(codes, vmins, vmaxs).astype(np.float32)
        q_eff = rfwht(q_feat, seed=RFWHT_SEED).astype(np.float32)
        extra["gallery_bytes"] = int(codes.nbytes + vmins.nbytes + vmaxs.nbytes)
        if verify:
            extra["reference_check"] = verify_against_reference(
                q_eff, codes, vmins, vmaxs, g_eff, pq.d
            )
    elif cfg["G"] == "float32_rfwht":
        g_eff = rfwht(g_feat, seed=RFWHT_SEED).astype(np.float32)
        q_eff = rfwht(q_feat, seed=RFWHT_SEED).astype(np.float32)
        extra["gallery_bytes"] = int(g_feat.astype(np.float32).nbytes)
    else:  # float32
        g_eff, q_eff = g_feat, q_feat
        extra["gallery_bytes"] = int(g_feat.astype(np.float32).nbytes)

    # ── distances, predictions, signals ──────────────────────────────────────
    t0 = time.perf_counter()
    dists = squared_l2(q_eff, g_eff)
    t_dist = time.perf_counter() - t0

    stats = GalleryStats.from_intra_class_nn(intra_class_nn(g_eff, g_lbl), n_classes)
    return {
        "dists": dists,
        "gallery_labels": g_lbl,
        "query_labels": q_lbl,
        "stats": stats,
        "n_classes": n_classes,
        "distance_time_s": t_dist,
        "extra": extra,
    }


def run_cell(cfg: dict, cache: dict, seed: int, k: int, verify: bool) -> dict:
    """Evaluate one factorial cell for one seed."""
    built = build_cell_arrays(cfg, cache, seed, verify)
    dists = built["dists"]
    g_lbl = built["gallery_labels"]
    q_lbl = built["query_labels"]
    stats = built["stats"]
    n_classes = built["n_classes"]
    t_dist = built["distance_time_s"]
    extra = built["extra"]

    scores, pred = compute_all(dists, g_lbl, stats, k=min(k, len(g_lbl)))
    correct = (pred == q_lbl).astype(np.int64)

    per_class = {
        int(c): float((pred[q_lbl == c] == c).mean())
        for c in range(n_classes)
        if (q_lbl == c).any()
    }

    result = {
        "config": cfg["name"],
        "factors": {kk: cfg[kk] for kk in ("A", "Q", "G", "S")},
        "seed": seed,
        "k": int(min(k, len(g_lbl))),
        "n_gallery": int(len(g_lbl)),
        "n_queries": int(len(q_lbl)),
        "n_classes": n_classes,
        "accuracy": float(correct.mean()),
        "macro_recall": float(np.mean(list(per_class.values()))),
        "per_class_recall": per_class,
        "tie_share_d1_d2": float(
            (np.sort(dists, axis=1)[:, 0] == np.sort(dists, axis=1)[:, 1]).mean()
        ),
        "distance_time_s": round(t_dist, 3),
        **extra,
        "signals": {},
        "signal_correlations": {},
    }

    # Rank correlations against the simple baselines, to catch a renamed margin.
    from scipy.stats import spearmanr

    base = {n: scores[n] for n in ("neg_d1", "margin_rel") if n in scores}
    for name, s in scores.items():
        finite = np.isfinite(s)
        result["signal_correlations"][name] = {
            bn: (
                float(spearmanr(s[finite & np.isfinite(b)], b[finite & np.isfinite(b)]).statistic)
                if (finite & np.isfinite(b)).sum() > 2
                else None
            )
            for bn, b in base.items()
        }

    # Signal evaluation: direction chosen on the tuning half, reported on the other.
    rng = np.random.default_rng(seed)
    idx = rng.permutation(correct.size)
    cut = correct.size // 2
    tune_all, eval_all = idx[:cut], idx[cut:]
    for name, s in scores.items():
        finite = np.isfinite(s)
        tune, ev = tune_all[finite[tune_all]], eval_all[finite[eval_all]]
        if tune.size == 0 or ev.size == 0 or correct[ev].sum() in (0, ev.size):
            result["signals"][name] = {"auroc": None, "reason": "degenerate split"}
            continue
        a_tune = auroc(s[tune], correct[tune])
        direction = 1.0 if (np.isnan(a_tune) or a_tune >= 0.5) else -1.0
        se, ce = direction * s[ev], correct[ev]
        acc_cov = risk_coverage(se, ce, COVERAGES)
        result["signals"][name] = {
            "auroc": auroc(se, ce),
            "auroc_undirected": auroc(s[ev], ce),
            "direction": direction,
            "aurc": aurc(se, ce),
            "aurc_random": aurc(np.zeros_like(se), ce),
            "aurc_oracle": aurc(ce.astype(float), ce),
            "accuracy_at_coverage": {
                f"{c:.2f}": float(a) for c, a in zip(COVERAGES, acc_cov)
            },
            "n_at_coverage": {
                f"{c:.2f}": int(max(1, round(c * ce.size))) for c in COVERAGES
            },
            "n_eval": int(ev.size),
            "n_dropped_nonfinite": int((~finite).sum()),
        }
    return result


def build_configs(full_source: bool = True) -> list[dict]:
    cfgs = []
    for a in ("none", "icoral"):
        for q in ("onnx_fp32", "onnx_int8"):
            for g in ("float32", "pq3bit"):
                for s in ("source", "target"):
                    bits = f"{int(a == 'icoral')}{int(q == 'onnx_int8')}{int(g == 'pq3bit')}{int(s == 'target')}"
                    cfgs.append({"name": f"s2_{bits}", "A": a, "Q": q, "G": g, "S": s})
    if full_source:
        cfgs += [
            {"name": "ref_fullsrc_fp32_pq", "A": "icoral", "Q": "onnx_fp32", "G": "pq3bit",
             "S": "source", "full_source": True},
            {"name": "ref_fullsrc_int8_pq", "A": "icoral", "Q": "onnx_int8", "G": "pq3bit",
             "S": "source", "full_source": True},
            {"name": "ref_fullsrc_fp32_f32", "A": "icoral", "Q": "onnx_fp32", "G": "float32",
             "S": "source", "full_source": True},
        ]
    cfgs.append(
        {"name": "ref_rfwht_nopq", "A": "icoral", "Q": "onnx_fp32", "G": "float32_rfwht",
         "S": "source"}
    )
    cfgs.append(
        {"name": "ref_torch_pq_src", "A": "icoral", "Q": "torch_fp32", "G": "pq3bit",
         "S": "source", "full_source": True}
    )
    return cfgs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=PROJECT / "results" / "s2_features")
    ap.add_argument("--out", type=Path, default=PROJECT / "results" / "s2_factorial")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SPLIT_SEEDS))
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these config names")
    ap.add_argument("--no-full-source", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cache = {p.stem: np.load(p) for p in args.features.glob("*.npy")}
    print(f"cached arrays: {sorted(cache)}")

    configs = build_configs(full_source=not args.no_full_source)
    if args.only:
        configs = [c for c in configs if c["name"] in args.only]

    t_start = time.perf_counter()
    done = 0
    for cfg in configs:
        for si, seed in enumerate(args.seeds):
            dst = args.out / f"{cfg['name']}_seed{seed}.json"
            if dst.exists():
                done += 1
                continue
            t = time.perf_counter()
            res = run_cell(cfg, cache, seed, args.k, verify=(si == 0))
            res["meta"] = {
                "command": " ".join(sys.argv),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "git": {
                    "edge-reliability-gate": git_hash(PROJECT),
                    "plantdiag-edge": git_hash(PLANTDIAG),
                },
                "features_dir": str(args.features),
                "wall_s": round(time.perf_counter() - t, 2),
                "environment": {
                    "hostname": socket.gethostname(),
                    "machine": platform.machine(),
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                },
            }
            dst.write_text(json.dumps(res, indent=2, ensure_ascii=False))
            done += 1
            cg = res["signals"].get("class_gap", {}).get("auroc")
            nd = res["signals"].get("neg_d1", {}).get("auroc_undirected")
            print(
                f"[{done}/{len(configs) * len(args.seeds)}] {cfg['name']} seed{seed}  "
                f"N={res['n_gallery']}  acc={res['accuracy']:.4f}  "
                f"neg_d1(undirected)={nd:.3f}  class_gap={cg:.3f}  "
                f"({res['meta']['wall_s']}s)",
                flush=True,
            )
    print(f"\n{done} cell-runs in {time.perf_counter() - t_start:.0f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
