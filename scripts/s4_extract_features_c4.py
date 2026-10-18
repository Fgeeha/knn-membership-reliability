#!/usr/bin/env python3
"""S4 step 0: cache ResNet-50 (C4) features for every image set of the S3 bridge.

The S3 bridge and two-loop experiments established their claims on MobileNetV2
(C6). This caches the same image sets through the second exported backbone of
the frozen plantdiag-edge study — ResNet-50 (C4, d=2048, ONNX fp32 and INT8) —
so the bridge/two-loop can be replicated with the *only* change being the
feature extractor (preregistered H5/H6, S4_PREREGISTRATION.md §2).

Image lists are taken verbatim from the S2/S3 caches (paths.json), so every set
is byte-for-byte the same images in the same order; labels are reused from the
existing *_labels.npy files.
"""

from __future__ import annotations

import argparse
import hashlib
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
sys.path.insert(0, str(PROJECT / "scripts"))

from s2_extract_features import extract_onnx  # noqa: E402

BUNDLE = PLANTDIAG / "artifacts_c4_pq"
MODELS = {
    "onnx_fp32": BUNDLE / "c4_resnet50_fp32.onnx",
    "onnx_int8": BUNDLE / "c4_resnet50_int8.onnx",
}
# (set name, paths.json source, labels source)
SETS = (
    ("src", "s2_features/src_paths.json", "s2_features/src_labels.npy"),
    ("tgt_tr", "s2_features/tgt_tr_paths.json", "s2_features/tgt_tr_labels.npy"),
    ("tgt_val", "s2_features/tgt_val_paths.json", "s2_features/tgt_val_labels.npy"),
    ("closed", "s3_features/closed_paths.json", "s3_features/closed_labels.npy"),
    ("ood_known_crop", "s3_features/ood_known_crop_paths.json", "s3_features/ood_known_crop_labels.npy"),
    ("ood_unseen_crop", "s3_features/ood_unseen_crop_paths.json", "s3_features/ood_unseen_crop_labels.npy"),
)


def git_hash(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=PROJECT / "results" / "s4_features_c4")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--sets", nargs="+", default=[s[0] for s in SETS])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    manifest: dict[str, dict] = {}
    for name, paths_rel, labels_rel in SETS:
        if name not in args.sets:
            continue
        files = [Path(p) for p in json.loads((PROJECT / "results" / paths_rel).read_text())]
        labels = np.load(PROJECT / "results" / labels_rel)
        assert len(files) == len(labels), (name, len(files), len(labels))
        missing = [p for p in files[:: max(1, len(files) // 50)] if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{name}: sampled paths missing, e.g. {missing[0]}")
        np.save(args.out / f"{name}_labels.npy", labels)
        (args.out / f"{name}_paths.json").write_text(json.dumps([str(p) for p in files]))
        manifest[name] = {"n_images": len(files)}
        for fset, model in MODELS.items():
            dst = args.out / f"{name}_{fset}.npy"
            if dst.exists():
                print(f"{name}_{fset}: cached", flush=True)
                continue
            print(f"{name}_{fset}: extracting {len(files)} images ...", flush=True)
            t1 = time.perf_counter()
            f = extract_onnx(files, model, args.batch)
            np.save(dst, f)
            dt = time.perf_counter() - t1
            print(f"  -> {f.shape} in {dt:.0f} s ({len(files) / dt:.0f} img/s)", flush=True)

    caches = {}
    for p in sorted(args.out.glob("*.npy")):
        a = np.load(p, mmap_mode="r")
        caches[p.name] = {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": sha256_file(p)}

    meta = {
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": {"edge-reliability-gate": git_hash(PROJECT), "plantdiag-edge": git_hash(PLANTDIAG)},
        "models": {k: {"path": str(v), "sha256": sha256_file(v)} for k, v in MODELS.items()},
        "image_lists": "verbatim from s2_features/s3_features paths.json (same order)",
        "preprocessing": "identical to s2_extract_features.extract_onnx",
        "groups": manifest,
        "caches": caches,
        "wall_s": round(time.perf_counter() - t0, 1),
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    (args.out / "s4_extract_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\ndone in {meta['wall_s']} s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
