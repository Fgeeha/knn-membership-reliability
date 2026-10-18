#!/usr/bin/env python3
"""S2 step 1: cache backbone features for every image of the factorial protocol.

The backbone is a deterministic function — `backbone.eval()`, all parameters
frozen, BatchNorm statistics fixed, and `.train()` is never called (verified by
reading `plantdiag-edge/scripts/eval_c4_c6_ablation.py`). So a feature vector
depends on the image alone, not on the seed. Extracting once and replaying seeds
over the cache is therefore not an approximation: every cell of the factorial
sees literally the same vectors, which is stronger than re-extracting per seed.

Three feature sets are produced, for three different jobs:

  `torch_fp32`  torchvision MobileNetV2 with the published `_TRANSFORM`
                (bilinear resize). Reproduces the published factorial cells.
  `onnx_fp32`   the deployed FP32 ONNX graph, PIL bicubic resize as
                `ONNXRunner` does it.
  `onnx_int8`   the deployed INT8 ONNX graph (static QDQ, per-channel).

`onnx_fp32` and `onnx_int8` differ **only** in weight precision, so their
contrast is a genuine quantization factor. The published torch path cannot
provide one: `quantize_dynamic` quantizes zero modules of an all-convolutional
backbone (see FACTORIAL_DESIGN.md §3.2), which this script re-verifies and
records rather than assumes.

The torch and ONNX sets are not bit-comparable to each other: torchvision
resizes bilinear, `ONNXRunner` bicubic. They are therefore never mixed inside a
single factorial cell.
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
sys.path.insert(0, str(PROJECT / "src"))

from scripts.train_cdan import (  # noqa: E402
    _TRANSFORM,
    _collect_source,
    _collect_target,
    load_config,
    load_labels,
    split_target_stratified,
)

SPLIT_SEED = 42  # fixed eval split of every published run
BUNDLE = PLANTDIAG / "artifacts_pq"


def git_hash(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def verify_quantize_dynamic_is_degenerate() -> dict:
    """Re-measure the claim that the published INT8 path quantizes nothing."""
    import torch
    import torch.nn as nn
    import torchvision.models as models

    out = {}
    for name, build in (
        ("mobilenet_v2", lambda: models.mobilenet_v2(weights=None).features),
        ("resnet50", lambda: nn.Sequential(*list(models.resnet50(weights=None).children())[:-2])),
    ):
        m = build().eval()
        q = torch.quantization.quantize_dynamic(m, {nn.Linear, nn.Conv2d}, dtype=torch.qint8).eval()
        size = lambda mm: sum(p.numel() * p.element_size() for p in mm.state_dict().values()) / 1e6
        n_q = sum(1 for _, mod in q.named_modules() if "quantized" in type(mod).__module__)
        torch.manual_seed(0)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            a = nn.AdaptiveAvgPool2d(1)(m(x)).flatten(1)
            b = nn.AdaptiveAvgPool2d(1)(q(x)).flatten(1)
        out[name] = {
            "quantized_modules": int(n_q),
            "size_fp32_mb": round(size(m), 3),
            "size_int8_mb": round(size(q), 3),
            "max_abs_feature_diff": float((a - b).abs().max()),
            "bitwise_identical": bool(torch.equal(a, b)),
        }
    return out


def extract_torch(paths: list[Path], batch: int, device: str) -> np.ndarray:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from scripts.train_cdan import FeatureExtractor

    class _DS(Dataset):
        def __len__(self) -> int:
            return len(paths)

        def __getitem__(self, i: int):
            return _TRANSFORM(Image.open(paths[i]).convert("RGB"))

    net = FeatureExtractor("mobilenet_v2", 1280, freeze=True).to(device).eval()
    loader = DataLoader(_DS(), batch_size=batch, num_workers=8, shuffle=False)
    feats, t0, done = [], time.perf_counter(), 0
    with torch.no_grad():
        for x in loader:
            feats.append(net(x.to(device)).cpu().numpy().astype(np.float32))
            done += len(x)
            if done % (batch * 20) == 0:
                print(f"    torch {done}/{len(paths)} ({done / (time.perf_counter() - t0):.0f}/s)", flush=True)
    return np.concatenate(feats)


def extract_onnx(paths: list[Path], model: Path, batch: int) -> np.ndarray:
    import onnxruntime as ort
    from PIL import Image

    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def prep(p: Path) -> np.ndarray:
        with Image.open(p) as im:
            # Same call ONNXRunner._preprocess makes, including PIL's default
            # resample filter — deliberately not "fixed" to match torchvision.
            img = im.convert("RGB").resize((224, 224))
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - mean) / std
        return arr.transpose(2, 0, 1)

    feats, t0 = [], time.perf_counter()
    for i in range(0, len(paths), batch):
        chunk = np.stack([prep(p) for p in paths[i : i + batch]])
        feats.append(sess.run(["features"], {"input": chunk})[0].astype(np.float32))
        if (i // batch) % 20 == 0 and i:
            print(f"    onnx {i}/{len(paths)} ({i / (time.perf_counter() - t0):.0f}/s)", flush=True)
    return np.concatenate(feats)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=PROJECT / "results" / "s2_features")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="cap images per split (smoke only)")
    ap.add_argument("--sets", nargs="+", default=["torch_fp32", "onnx_fp32", "onnx_int8"])
    args = ap.parse_args()

    import torch

    device = args.device if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    cfg = load_config(str(PLANTDIAG / "configs/cdan_train.yaml"))
    pairs, target_labels = load_labels(str(PLANTDIAG / cfg["data"]["labels_file"]))
    l2i = {lbl: i for i, lbl in enumerate(target_labels)}

    src = _collect_source(pairs, Path(cfg["data"]["source_root"]), l2i)
    tgt = _collect_target(pairs, Path(cfg["data"]["target_root"]), l2i)
    tgt_tr, tgt_val = split_target_stratified(tgt, val_ratio=0.2, seed=SPLIT_SEED)
    splits = {"src": src, "tgt_tr": tgt_tr, "tgt_val": tgt_val}
    if args.limit:
        splits = {k: v[: args.limit] for k, v in splits.items()}

    for name, samples in splits.items():
        np.save(args.out / f"{name}_labels.npy", np.array([s.label_idx for s in samples], dtype=np.int32))
        (args.out / f"{name}_paths.json").write_text(json.dumps([str(s.path) for s in samples]))
        print(f"{name}: {len(samples)} images")

    timing = {}
    for fset in args.sets:
        for name, samples in splits.items():
            dst = args.out / f"{name}_{fset}.npy"
            if dst.exists():
                print(f"  {dst.name} exists, skipping")
                continue
            paths = [s.path for s in samples]
            print(f"  extracting {fset} for {name} ({len(paths)} images) ...", flush=True)
            t = time.perf_counter()
            if fset == "torch_fp32":
                f = extract_torch(paths, args.batch, device)
            elif fset == "onnx_fp32":
                f = extract_onnx(paths, BUNDLE / "c6_mobilenet_v2_fp32.onnx", args.batch)
            elif fset == "onnx_int8":
                f = extract_onnx(paths, BUNDLE / "c6_mobilenet_v2_int8.onnx", args.batch)
            else:
                raise ValueError(f"unknown feature set {fset}")
            np.save(dst, f)
            timing[f"{fset}/{name}"] = round(time.perf_counter() - t, 1)
            print(f"    -> {f.shape} in {timing[f'{fset}/{name}']} s")

    # Cross-set agreement, measured rather than assumed.
    agreement = {}
    val_sets = {s: args.out / f"tgt_val_{s}.npy" for s in args.sets}
    if val_sets.get("onnx_fp32", Path()).exists() and val_sets.get("onnx_int8", Path()).exists():
        a = np.load(val_sets["onnx_fp32"])
        b = np.load(val_sets["onnx_int8"])
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
        agreement["onnx_fp32_vs_onnx_int8"] = {
            "cosine_mean": float(cos.mean()),
            "cosine_min": float(cos.min()),
            "identical": bool(np.array_equal(a, b)),
            "note": "genuine quantization factor: the two graphs must NOT be identical",
        }
    if val_sets.get("torch_fp32", Path()).exists() and val_sets.get("onnx_fp32", Path()).exists():
        a = np.load(val_sets["torch_fp32"])
        b = np.load(val_sets["onnx_fp32"])
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
        agreement["torch_fp32_vs_onnx_fp32"] = {
            "cosine_mean": float(cos.mean()),
            "cosine_min": float(cos.min()),
            "note": "same weights, different resize filter (bilinear vs bicubic); never mixed in one cell",
        }

    meta = {
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "split_seed": SPLIT_SEED,
        "git": {"edge-reliability-gate": git_hash(PROJECT), "plantdiag-edge": git_hash(PLANTDIAG)},
        "protocol": {
            "source_root": cfg["data"]["source_root"],
            "target_root": cfg["data"]["target_root"],
            "labels_file": cfg["data"]["labels_file"],
            "n_classes": len(target_labels),
            "target_labels": target_labels,
            "counts": {k: len(v) for k, v in splits.items()},
        },
        "models": {
            "onnx_fp32": str(BUNDLE / "c6_mobilenet_v2_fp32.onnx"),
            "onnx_int8": str(BUNDLE / "c6_mobilenet_v2_int8.onnx"),
            "onnx_int8_sha256": hashlib.sha256(
                (BUNDLE / "c6_mobilenet_v2_int8.onnx").read_bytes()
            ).hexdigest(),
        },
        "quantize_dynamic_degeneracy": verify_quantize_dynamic_is_degenerate(),
        "feature_set_agreement": agreement,
        "timing_s": timing,
        "total_s": round(time.perf_counter() - t_start, 1),
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    try:
        import onnxruntime

        meta["environment"]["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        pass
    (args.out / "extract_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print("\nquantize_dynamic degeneracy check:")
    for k, v in meta["quantize_dynamic_degeneracy"].items():
        print(f"  {k}: {v['quantized_modules']} quantized modules, "
              f"{v['size_fp32_mb']} -> {v['size_int8_mb']} MB, identical={v['bitwise_identical']}")
    for k, v in agreement.items():
        print(f"{k}: cosine mean {v['cosine_mean']:.4f} min {v['cosine_min']:.4f}")
    print(f"\ndone in {meta['total_s']} s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
