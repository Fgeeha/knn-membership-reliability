#!/usr/bin/env python3
"""S3 (bounded): can the reliability signals refuse data outside the label space?

Hypothesis H3 of `NEXT_RESEARCH_DECISION.md` was registered before any S2 result
and asks two things. The first half — transfer to a second field domain — needs a
new label alignment and is deferred. The second half needs **no alignment at
all**: an image of a crop the system has never been given a class for should be
refused, and whether the signals support that refusal is answerable directly.

Three tiers of "outside", ordered by how far they sit from the training space:

  `beans`        leaves, but of a crop absent from the 23-class space.
                 Same organ, different crop.
  `fcdd_apple`   apple **fruit**. The crop is in the label space, the organ is
                 not. Same crop, different organ.
  `fcdd_other`   guava, mango and pomegranate fruit. Neither crop nor organ is
                 in the label space — the far tier.

The task here is **not** the one S2 measured. S2 asked "is this prediction
correct"; this asks "is this input inside the label space at all". The two are
reported separately and never averaged together.

A signal is scored by AUROC for separating PlantDoc-val (in-distribution) from
each OOD tier, with the direction taken as "higher signal = in-distribution".
No direction is fitted, because there is nothing to fit it on in deployment: a
system that needs OOD labels to configure its OOD detector is not deployable.
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
ECCV = PROJECT.parent / "plant-disease-eccv2026"
sys.path.insert(0, str(PLANTDIAG))
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from reliability_gate.metrics import auroc, bootstrap_ci  # noqa: E402
from reliability_gate.signals import compute_all  # noqa: E402
from s2_extract_features import extract_onnx  # noqa: E402
from s2_run_factorial import build_cell_arrays, build_configs  # noqa: E402

IMG = {".jpg", ".jpeg", ".png", ".bmp"}
TIERS = {
    "beans": (ECCV / "data" / "external" / "beans", "leaves of a crop outside the label space"),
    "fcdd_apple": (ECCV / "data" / "FCDD_raw" / "APPLE", "apple fruit: crop in space, organ not"),
    "fcdd_other": (None, "guava, mango, pomegranate fruit: neither crop nor organ in space"),
}
FCDD_OTHER = ["GUAVA", "MANGO", "POMEGRANATE"]


def git_hash(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def collect(root: Path, limit: int, seed: int) -> list[Path]:
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG)
    if limit and len(files) > limit:
        idx = np.sort(np.random.default_rng(seed).choice(len(files), limit, replace=False))
        files = [files[i] for i in idx]
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=PROJECT / "results" / "s2_features")
    ap.add_argument("--out", type=Path, default=PROJECT / "results" / "s3_ood")
    ap.add_argument("--config", default="s2_1011", help="pipeline cell providing the gallery")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 42])
    ap.add_argument("--limit", type=int, default=1200, help="cap images per tier")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    cfg = {c["name"]: c for c in build_configs()}[args.config]
    if cfg["Q"] != "onnx_fp32":
        raise SystemExit(f"config {args.config} uses {cfg['Q']}; OOD features are extracted with onnx_fp32")

    # ── OOD features, cached so reruns are free ──────────────────────────────
    tiers: dict[str, list[Path]] = {}
    for name, (root, _) in TIERS.items():
        if name == "fcdd_other":
            files: list[Path] = []
            for crop in FCDD_OTHER:
                files += collect(ECCV / "data" / "FCDD_raw" / crop, 0, args.seeds[0])
            if args.limit and len(files) > args.limit:
                idx = np.sort(np.random.default_rng(args.seeds[0]).choice(len(files), args.limit, replace=False))
                files = [files[i] for i in idx]
        else:
            files = collect(root, args.limit, args.seeds[0])
        tiers[name] = files
        print(f"{name}: {len(files)} images")

    ood_feat: dict[str, np.ndarray] = {}
    for name, files in tiers.items():
        dst = args.out / f"{name}_onnx_fp32.npy"
        if dst.exists():
            ood_feat[name] = np.load(dst)
            print(f"  {dst.name} cached")
            continue
        print(f"  extracting {name} ...", flush=True)
        f = extract_onnx(files, PLANTDIAG / "artifacts_pq" / "c6_mobilenet_v2_fp32.onnx", args.batch)
        np.save(dst, f)
        ood_feat[name] = f

    cache = {p.stem: np.load(p) for p in args.features.glob("*.npy")}
    results: dict[str, dict] = {}

    for seed in args.seeds:
        built = build_cell_arrays(cfg, cache, seed)
        g_lbl = built["gallery_labels"]
        stats = built["stats"]
        n_classes = built["n_classes"]

        # In-distribution scores come from the very arrays S2 evaluated.
        in_scores, _ = compute_all(built["dists"], g_lbl, stats, k=args.k)

        # OOD queries traverse the identical path. The gallery is rebuilt inside
        # `build_cell_arrays`, so the only thing that changes is the query set;
        # the iCORAL transform is refitted on target-train exactly as there.
        for tier, feats in ood_feat.items():
            d_ood = ood_distances(cfg, cache, seed, feats)
            o_scores, _ = compute_all(d_ood, g_lbl, stats, k=args.k)
            for sig in in_scores:
                a, b = in_scores[sig], o_scores[sig]
                m = np.isfinite(a)
                mo = np.isfinite(b)
                s = np.concatenate([a[m], b[mo]])
                y = np.concatenate([np.ones(m.sum(), int), np.zeros(mo.sum(), int)])
                results.setdefault(tier, {}).setdefault(sig, []).append(auroc(s, y))

    summary = {}
    for tier, sigs in results.items():
        summary[tier] = {}
        for sig, vals in sigs.items():
            v = np.asarray(vals, float)
            lo, hi = bootstrap_ci(v) if v.size > 1 else (float(v[0]), float(v[0]))
            summary[tier][sig] = {
                "auroc_in_vs_ood": float(v.mean()),
                "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                "ci95": [lo, hi],
                "all_above_half": bool((v > 0.5).all()),
                "per_seed": v.tolist(),
            }

    payload = {
        "task": "in-distribution (PlantDoc val) vs outside the label space",
        "note": "different task from S2 correctness discrimination; never averaged with it",
        "config": args.config,
        "factors": {kk: cfg[kk] for kk in ("A", "Q", "G", "S")},
        "k": args.k,
        "tiers": {
            name: {"n_images": len(files), "meaning": TIERS[name][1]}
            for name, files in tiers.items()
        },
        "n_in_distribution": int(len(cache["tgt_val_labels"])),
        "seed_variability": (
            "a target-domain gallery is the whole target-train set and is not resampled, "
            "so all seeds coincide and the measurement has one replicate; a source-domain "
            "gallery is stratified-subsampled per seed and does vary"
        ),
        "summary": summary,
        "meta": {
            "command": " ".join(sys.argv),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git": {"edge-reliability-gate": git_hash(PROJECT), "plantdiag-edge": git_hash(PLANTDIAG)},
            "seeds": args.seeds,
            "wall_s": round(time.perf_counter() - t_start, 1),
            "environment": {
                "hostname": socket.gethostname(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
        },
    }
    (args.out / "ood_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"\ngallery from {args.config} (S={cfg['S']}, G={cfg['G']}), k={args.k}, "
          f"{len(args.seeds)} seeds, in-distribution N={payload['n_in_distribution']}")
    sigs = sorted(next(iter(summary.values())))
    print(f"\n{'tier':<14}{'N':>6}" + "".join(f"{s[:11]:>13}" for s in sigs))
    for tier in ("beans", "fcdd_apple", "fcdd_other"):
        if tier not in summary:
            continue
        row = f"{tier:<14}{len(tiers[tier]):>6}"
        for s in sigs:
            row += f"{summary[tier][s]['auroc_in_vs_ood']:>13.3f}"
        print(row)
    print(f"\nwritten -> {args.out / 'ood_results.json'}  ({payload['meta']['wall_s']} s)")
    return 0


def ood_distances(cfg: dict, cache: dict, seed: int, feats: np.ndarray) -> np.ndarray:
    """Distances from OOD queries to the same gallery the S2 cell used.

    Implemented by temporarily substituting the evaluation split in the cache, so
    the query passes through byte-identical adaptation, rotation and quantization
    code rather than a parallel reimplementation.
    """
    shim = dict(cache)
    shim["tgt_val_onnx_fp32"] = feats.astype(np.float32)
    shim["tgt_val_labels"] = np.zeros(len(feats), dtype=np.int32)
    return build_cell_arrays(cfg, shim, seed)["dists"]


if __name__ == "__main__":
    raise SystemExit(main())
