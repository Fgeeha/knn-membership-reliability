#!/usr/bin/env python3
"""S3 step 1: cache PlantWild features and build the closed-set / OOD splits.

Only PlantWild needs new features: the PlantVillage and PlantDoc galleries reuse
the S2 cache byte for byte, which is what keeps the two stages comparable.

Splits, all derived from the alignment file and never from a result:

  `closed`  the 13 conservatively aligned classes, split 80/20 with a fixed seed.
            The 80 % part supplies the domain-matched gallery, the 20 % part is
            the query set for every cell of the factorial.
  `ood_known_crop`   31 classes whose crop is in the label space but whose
            disease is not.
  `ood_unseen_crop`  69 classes whose crop is absent from the label space.
  `author_pending`   the 2 classes awaiting a human decision, extracted but
            excluded from every headline number.

Every cache entry carries the model hash, preprocessing identity, split id,
alignment hash, dtype, shape and a sha256 of the array, so a later run can prove
it is looking at the same numbers.
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

PLANTWILD_ROOT = Path(
    "data/plantwild_v2"
)
SPLIT_SEED = 42
VAL_RATIO = 0.2
BUNDLE = PLANTDIAG / "artifacts_pq"


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


def stratified_split(files: list[Path], labels: list[int], seed: int, ratio: float):
    """Per-class 80/20 split, same shape of rule as `split_target_stratified`."""
    import random

    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for i, y in enumerate(labels):
        by_class.setdefault(y, []).append(i)
    tr, va = [], []
    for y in sorted(by_class):
        idx = list(by_class[y])
        rng.shuffle(idx)
        n_val = max(1, round(len(idx) * ratio))
        va += idx[:n_val]
        tr += idx[n_val:]
    return sorted(tr), sorted(va)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alignment", type=Path, default=PROJECT / "label_alignment.json")
    ap.add_argument("--plantwild", type=Path, default=PLANTWILD_ROOT)
    ap.add_argument("--out", type=Path, default=PROJECT / "results" / "s3_features")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--sets", nargs="+", default=["onnx_fp32", "onnx_int8"])
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    align = json.loads(args.alignment.read_text())
    spec = json.loads((PLANTDIAG / "configs" / "labels_clean.json").read_text())
    canonical = list(spec["target_labels"])
    l2i = {l: i for i, l in enumerate(canonical)}

    groups = {"closed": [], "ood_known_crop": [], "ood_unseen_crop": [], "author_pending": []}
    action_to_group = {
        "INCLUDE_CONSERVATIVE": "closed",
        "OOD_KNOWN_CROP": "ood_known_crop",
        "OOD_UNSEEN_CROP": "ood_unseen_crop",
        "AUTHOR_DECISION": "author_pending",
    }
    for r in align["rows"]:
        groups[action_to_group[r["action"]]].append(r)

    manifest: dict[str, dict] = {}
    for gname, rows in groups.items():
        files: list[Path] = []
        labels: list[int] = []
        src_class: list[str] = []
        for r in rows:
            d = args.plantwild / r["external_label"]
            imgs = sorted(p for p in d.iterdir() if p.suffix.lower() == ".jpg")
            files += imgs
            y = l2i[r["canonical_label"]] if r["canonical_label"] else -1
            labels += [y] * len(imgs)
            src_class += [r["external_label"]] * len(imgs)
        if not files:
            continue
        lab = np.asarray(labels, dtype=np.int32)
        np.save(args.out / f"{gname}_labels.npy", lab)
        (args.out / f"{gname}_paths.json").write_text(json.dumps([str(p) for p in files]))
        (args.out / f"{gname}_srcclass.json").write_text(json.dumps(src_class))
        manifest[gname] = {"n_images": len(files), "n_classes": len(rows)}
        print(f"{gname}: {len(files)} images, {len(rows)} external classes")

        if gname == "closed":
            tr, va = stratified_split(files, labels, SPLIT_SEED, VAL_RATIO)
            np.save(args.out / "closed_train_idx.npy", np.asarray(tr, dtype=np.int32))
            np.save(args.out / "closed_query_idx.npy", np.asarray(va, dtype=np.int32))
            manifest[gname]["n_gallery_pool"] = len(tr)
            manifest[gname]["n_query"] = len(va)
            per_class = {int(c): int((lab[tr] == c).sum()) for c in np.unique(lab)}
            manifest[gname]["gallery_pool_per_class"] = per_class
            manifest[gname]["min_gallery_pool_per_class"] = min(per_class.values())
            print(f"  gallery pool {len(tr)}, queries {len(va)}, "
                  f"min per class in pool {min(per_class.values())}")

        for fset in args.sets:
            dst = args.out / f"{gname}_{fset}.npy"
            if dst.exists():
                print(f"  {dst.name} cached")
                continue
            model = BUNDLE / (
                "c6_mobilenet_v2_fp32.onnx" if fset == "onnx_fp32" else "c6_mobilenet_v2_int8.onnx"
            )
            print(f"  extracting {fset} ({len(files)} images) ...", flush=True)
            f = extract_onnx(files, model, args.batch)
            np.save(dst, f)
            print(f"    -> {f.shape}")

    caches = {}
    for p in sorted(args.out.glob("*.npy")):
        a = np.load(p, mmap_mode="r")
        caches[p.name] = {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "sha256": sha256_file(p),
        }

    meta = {
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": {"edge-reliability-gate": git_hash(PROJECT), "plantdiag-edge": git_hash(PLANTDIAG)},
        "alignment": {
            "version": align["mapping_version"],
            "sha256": align["mapping_sha256"],
            "file": str(args.alignment),
        },
        "split": {"seed": SPLIT_SEED, "val_ratio": VAL_RATIO, "rule": "per-class stratified"},
        "preprocessing": {
            "resize": "PIL Image.resize((224,224)) with library default filter",
            "scale": "x/255",
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "identical_to": "s2_extract_features.extract_onnx",
        },
        "models": {
            fset: {
                "path": str(BUNDLE / ("c6_mobilenet_v2_fp32.onnx" if fset == "onnx_fp32" else "c6_mobilenet_v2_int8.onnx")),
                "sha256": sha256_file(BUNDLE / ("c6_mobilenet_v2_fp32.onnx" if fset == "onnx_fp32" else "c6_mobilenet_v2_int8.onnx")),
            }
            for fset in args.sets
        },
        "groups": manifest,
        "caches": caches,
        "canonical_space": {"n_classes": len(canonical), "labels": canonical},
        "wall_s": round(time.perf_counter() - t0, 1),
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    (args.out / "s3_extract_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\ndone in {meta['wall_s']} s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
