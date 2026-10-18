#!/usr/bin/env python3
"""Measure the deployment cost of the reliability gate.

Hypothesis H2 of `NEXT_RESEARCH_DECISION.md` states the gate must add no more
than 10 % to the C6 bundle (2.92 MB) and 10 % to its end-to-end latency
(103.32 ms p50). This measures both.

What is measured is the **marginal** cost: the gate is fed the distance vector
the kNN stage has already computed, because that vector is not extra work — the
retrieval head produces it whether or not a gate consumes it. Timing the
distance computation again would inflate the gate's cost with the pipeline's.

Depends on numpy only, so it runs unchanged on the Raspberry Pi without
installing anything.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEPLOY_MB = 2.924  # RESULTS_INDEX.md position 1, measured 2026-08-13
E2E_P50_MS = 103.32  # RESULTS_INDEX.md, rpi4_c6_e2e_v2_20260813


def class_gap(d: np.ndarray, gal_labels: np.ndarray, n_classes: int, k: int) -> float:
    """Reference implementation of the winning signal, deployment form.

    Kept separate from `reliability_gate.signals` on purpose: this is what an
    edge build would actually contain — one query, no batching, no imports
    beyond numpy — and timing the vectorised research version would understate
    the per-query cost.
    """
    order = np.argsort(d, kind="mergesort")
    sd, sl = d[order], gal_labels[order]
    pred = sl[0]
    own = sd[sl == pred][:k]
    best = np.inf
    for c in range(n_classes):
        if c == pred:
            continue
        cd = sd[sl == c][:k]
        if cd.size:
            best = min(best, cd.mean())
    own_m = own.mean()
    return float((best - own_m) / (own_m + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-gallery", type=int, default=272)
    ap.add_argument("--n-classes", type=int, default=28)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    gal_labels = np.repeat(
        np.arange(args.n_classes), args.n_gallery // args.n_classes + 1
    )[: args.n_gallery].astype(np.int32)
    queries = rng.gamma(9.0, 30.0, size=(args.runs + args.warmup, args.n_gallery)).astype(np.float32)

    for i in range(args.warmup):
        class_gap(queries[i], gal_labels, args.n_classes, args.k)

    times = np.empty(args.runs)
    for i in range(args.runs):
        q = queries[args.warmup + i]
        t0 = time.perf_counter()
        class_gap(q, gal_labels, args.n_classes, args.k)
        times[i] = (time.perf_counter() - t0) * 1000.0

    # Gate state on disk: mu + sigma + a 19-point quantile grid per class, float32.
    stats_bytes = args.n_classes * (2 + 19) * 4
    p50 = float(np.percentile(times, 50))
    p95 = float(np.percentile(times, 95))

    res = {
        "signal": "class_gap",
        "setting": {
            "n_gallery": args.n_gallery,
            "n_classes": args.n_classes,
            "k": args.k,
            "runs": args.runs,
            "warmup": args.warmup,
            "seed": args.seed,
            "note": "marginal cost only; the kNN distance vector is taken as given",
        },
        "latency_ms": {
            "p50": p50,
            "p90": float(np.percentile(times, 90)),
            "p95": p95,
            "p99": float(np.percentile(times, 99)),
            "mean": float(times.mean()),
            "std": float(times.std(ddof=1)),
            "min": float(times.min()),
            "max": float(times.max()),
        },
        "memory_bytes": {
            "gallery_statistics": stats_bytes,
            "note": "mu, sigma and a 19-quantile grid per class, float32",
        },
        "budget_check": {
            "deploy_mb_reference": DEPLOY_MB,
            "e2e_p50_ms_reference": E2E_P50_MS,
            "added_size_fraction": stats_bytes / (DEPLOY_MB * 1e6),
            "added_latency_fraction_p50": p50 / E2E_P50_MS,
            "within_10pct_size": stats_bytes / (DEPLOY_MB * 1e6) < 0.10,
            "within_10pct_latency": p50 / E2E_P50_MS < 0.10,
        },
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    print(json.dumps(res, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2))
        print(f"\nwritten -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
