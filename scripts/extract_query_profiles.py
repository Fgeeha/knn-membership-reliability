#!/usr/bin/env python3
"""Extract full query-to-gallery distance profiles for the deployed C6 bundle.

Why this script exists at all: the per-sample dumps already published by
`plantdiag-edge` record only `d1` and `d2`. Every candidate difficulty signal in
this project needs the *whole* distance vector plus the gallery's own internal
geometry, and neither can be recovered from those dumps. This script produces
them once; everything downstream is pure analysis over its output.

It reproduces the deployed retrieval path of PlantDiagEdge configuration C6
without re-implementing it: the ONNX runner, the iCORAL adapter, the gallery
store and the asymmetric PolarQuant distance are all imported from
`plantdiag-edge` (provenance recorded in the run manifest).

Protocol
--------
* Gallery: the 272 PolarQuant-coded entries shipped in `artifacts_pq/gallery/`.
* Queries: every PlantDoc image that is **not** a gallery entry. The deployed
  gallery was drawn from both PlantDoc partitions, so excluding it by path is
  the only way to keep the query set held out.
* iCORAL is fitted on the gallery images' features only. No query image and no
  label of any kind enters the transform.

Output (one directory per run):
    distances.npy      float32 (Q, N)  query-to-gallery distances
    gallery_labels.npy int32   (N,)    class id per gallery entry
    gallery_gg.npy     float32 (N, N)  gallery-to-gallery distances
    query_labels.npy   int32   (Q,)    true class id per query
    query_paths.json           (Q,)    source image path per query
    classes.json                       label id -> label name
    run_meta.json                      command, seeds, git, env, hardware, timing
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
PLANTDIAG = PROJECT.parent / "plantdiag-edge"
sys.path.insert(0, str(PLANTDIAG))
sys.path.insert(0, str(PROJECT / "src"))

from plantdiag_edge.domain_adapter import IncrementalCORAL  # noqa: E402
from plantdiag_edge.edge_runtime import ONNXRunner  # noqa: E402
from plantdiag_edge.polarquant import rfwht, unpack_codes  # noqa: E402
from plantdiag_edge.polarquant.decoder import asymmetric_l2  # noqa: E402
from plantdiag_edge.polarquant.packing import _BITS  # noqa: E402

DEFAULT_BUNDLE = PLANTDIAG / "artifacts_pq"
DEFAULT_COV = PLANTDIAG / "artifacts"
DEFAULT_PLANTDOC = PROJECT.parent / "plant-disease-eccv2026" / "data" / "PlantDoc_raw"


def git_hash(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        return "unknown"


def load_gallery(bundle: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Read the deployed gallery: packed codes, per-vector scales, labels, paths."""
    db = bundle / "gallery" / "gallery.db"
    rows = sqlite3.connect(db).execute(
        "SELECT label, image_path, feature_file, vmin, vmax FROM entries ORDER BY id"
    ).fetchall()
    codes, vmins, vmaxs, labels, paths = [], [], [], [], []
    for label, image_path, feature_file, vmin, vmax in rows:
        codes.append(np.load(bundle / feature_file))
        vmins.append(vmin)
        vmaxs.append(vmax)
        labels.append(label)
        paths.append(image_path)
    return (
        np.stack(codes).astype(np.uint8),
        np.asarray(vmins, dtype=np.float32),
        np.asarray(vmaxs, dtype=np.float32),
        labels,
        paths,
    )


def bundle_path_to_local(bundle_path: str, plantdoc_root: Path) -> Path:
    """Map a deploy-bundle image path to its location in the local PlantDoc tree.

    The bundle was built from a flattened `/data/plantdoc/all/<CLASS>/<part>_<name>`
    tree; locally the same file lives at `<root>/<part>/<CLASS>/<name>`, where
    `<name>` itself still starts with the partition prefix.
    """
    parts = Path(bundle_path).parts
    cls, fname = parts[-2], parts[-1]
    partition, rest = fname.split("_", 1)
    return plantdoc_root / partition / cls / rest


def collect_queries(
    plantdoc_root: Path, gallery_paths: list[str], classes: list[str]
) -> tuple[list[Path], np.ndarray]:
    """Every PlantDoc image outside the gallery, restricted to gallery classes."""
    excluded = {bundle_path_to_local(p, plantdoc_root).resolve() for p in gallery_paths}
    cls2idx = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for partition in ("train", "test"):
        for cls_dir in sorted((plantdoc_root / partition).iterdir()):
            if not cls_dir.is_dir() or cls_dir.name not in cls2idx:
                continue
            for img in sorted(cls_dir.iterdir()):
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                if img.resolve() in excluded:
                    continue
                paths.append(img)
                labels.append(cls2idx[cls_dir.name])
    return paths, np.asarray(labels, dtype=np.int32)


def extract_features(runner: ONNXRunner, paths: list[Path], log_every: int = 250) -> np.ndarray:
    from PIL import Image

    feats = []
    t0 = time.perf_counter()
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            out = runner.predict(im.convert("RGB"))
        f = out.get("features", out.get("output", next(iter(out.values()))))
        feats.append(np.asarray(f, dtype=np.float32).ravel())
        if log_every and (i + 1) % log_every == 0:
            rate = (i + 1) / (time.perf_counter() - t0)
            print(f"  {i + 1}/{len(paths)} images  ({rate:.1f}/s)", flush=True)
    return np.stack(feats)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--cov-dir", type=Path, default=DEFAULT_COV)
    ap.add_argument("--plantdoc", type=Path, default=DEFAULT_PLANTDOC)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tag", default="c6_deploy")
    ap.add_argument("--seed", type=int, default=42, help="RFWHT seed; must match the bundle")
    ap.add_argument("--limit", type=int, default=0, help="cap query count (smoke runs only)")
    ap.add_argument(
        "--da", choices=["icoral", "none"], default="icoral",
        help="domain-adaptation stage; 'none' isolates the retrieval head",
    )
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = args.out or PROJECT / "results" / f"profiles_{args.tag}_{args.da}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    print(f"[1/6] gallery <- {args.bundle}", flush=True)
    codes, vmins, vmaxs, gal_label_names, gal_paths = load_gallery(args.bundle)
    classes = sorted(set(gal_label_names))
    cls2idx = {c: i for i, c in enumerate(classes)}
    gal_labels = np.asarray([cls2idx[c] for c in gal_label_names], dtype=np.int32)
    p_dim = codes.shape[1] * 8 // _BITS
    codes_q = unpack_codes(codes, p_dim)
    print(f"      {len(codes)} entries, {len(classes)} classes, padded dim {p_dim}")

    print(f"[2/6] query set <- {args.plantdoc}", flush=True)
    q_paths, q_labels = collect_queries(args.plantdoc, gal_paths, classes)
    if args.limit:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(len(q_paths), size=min(args.limit, len(q_paths)), replace=False))
        q_paths = [q_paths[i] for i in keep]
        q_labels = q_labels[keep]
    print(f"      {len(q_paths)} queries (gallery images excluded)")

    onnx_path = args.bundle / "c6_mobilenet_v2_int8.onnx"
    print(f"[3/6] features <- {onnx_path.name}", flush=True)
    runner = ONNXRunner(str(onnx_path), input_size=(224, 224))
    t = time.perf_counter()
    q_feat = extract_features(runner, q_paths)
    t_query_feat = time.perf_counter() - t
    gal_local = [bundle_path_to_local(p, args.plantdoc) for p in gal_paths]
    gal_feat = extract_features(runner, gal_local, log_every=0)

    print(f"[4/6] domain adaptation: {args.da}", flush=True)
    if args.da == "icoral":
        source_cov = np.load(args.cov_dir / "source_cov.npy")
        sqrt_src = np.load(args.cov_dir / "source_cov_sqrt.npy")
        icoral = IncrementalCORAL(q_feat.shape[1], sqrt_source_cov=sqrt_src)
        # Fitted on gallery features only: no query image informs the transform.
        for i in range(0, len(gal_feat), 16):
            icoral.update(gal_feat[i : i + 16].astype(np.float32))
        q_feat = icoral.align_to_source(q_feat.astype(np.float32), source_cov)
        gal_feat_aligned = icoral.align_to_source(gal_feat.astype(np.float32), source_cov)
    else:
        gal_feat_aligned = gal_feat

    print("[5/6] distance profiles", flush=True)
    t = time.perf_counter()
    dists = np.empty((len(q_feat), len(codes_q)), dtype=np.float32)
    for i, f in enumerate(q_feat):
        q_wht = rfwht(f[None, :], seed=args.seed)[0].astype(np.float32)
        dists[i] = asymmetric_l2(q_wht, codes_q, vmins, vmaxs)
    t_dist = time.perf_counter() - t

    # Gallery-to-gallery distances use the same asymmetric measure, so the
    # reference statistics live in the space the queries are actually scored in.
    gg = np.empty((len(codes_q), len(codes_q)), dtype=np.float32)
    for i, f in enumerate(gal_feat_aligned):
        g_wht = rfwht(f[None, :], seed=args.seed)[0].astype(np.float32)
        gg[i] = asymmetric_l2(g_wht, codes_q, vmins, vmaxs)

    print(f"[6/6] writing -> {out}", flush=True)
    np.save(out / "distances.npy", dists)
    np.save(out / "gallery_labels.npy", gal_labels)
    np.save(out / "gallery_gg.npy", gg)
    np.save(out / "query_labels.npy", q_labels)
    (out / "query_paths.json").write_text(json.dumps([str(p) for p in q_paths], indent=1))
    (out / "classes.json").write_text(json.dumps(classes, indent=1))

    pred = gal_labels[np.argmin(dists, axis=1)]
    acc = float((pred == q_labels).mean())
    per_class = [
        float((pred[q_labels == c] == c).mean()) for c in range(len(classes)) if (q_labels == c).any()
    ]
    meta = {
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "da": args.da,
        "git": {
            "edge-reliability-gate": git_hash(PROJECT),
            "plantdiag-edge": git_hash(PLANTDIAG),
        },
        "provenance": {
            "bundle": str(args.bundle),
            "onnx": str(onnx_path),
            "gallery_db": str(args.bundle / "gallery" / "gallery.db"),
            "source_cov": str(args.cov_dir / "source_cov.npy") if args.da == "icoral" else None,
            "dataset_root": str(args.plantdoc),
            "reused_from_plantdiag_edge": [
                "edge_runtime.ONNXRunner",
                "domain_adapter.IncrementalCORAL",
                "polarquant.rfwht",
                "polarquant.decoder.asymmetric_l2",
                "polarquant.packing.unpack_codes",
            ],
        },
        "data": {
            "n_gallery": int(len(codes)),
            "n_queries": int(len(q_paths)),
            "n_classes": len(classes),
            "feature_dim": int(q_feat.shape[1]),
            "padded_dim": int(p_dim),
        },
        "sanity": {
            "top1_accuracy": acc,
            "macro_recall": float(np.mean(per_class)),
            "tie_share_d1_d2": float(
                (np.sort(dists, axis=1)[:, 0] == np.sort(dists, axis=1)[:, 1]).mean()
            ),
        },
        "timing_s": {
            "query_features": round(t_query_feat, 2),
            "distances": round(t_dist, 2),
            "total": round(time.perf_counter() - t_start, 2),
        },
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    try:
        import onnxruntime

        meta["environment"]["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        pass
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"\ntop-1 accuracy {acc:.4f}   macro-recall {np.mean(per_class):.4f}")
    print(f"tie share (d1 == d2) {meta['sanity']['tie_share_d1_d2']:.4f}")
    print(f"done in {meta['timing_s']['total']:.1f} s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
