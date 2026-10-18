#!/usr/bin/env python3
"""Evaluate candidate difficulty signals on extracted distance profiles.

Protocol (mirrors ITIDS'2026 so the numbers are comparable with the published
ones): the query set is split in half; the **direction** of every signal is
chosen on the tuning half and the reported figures come from the evaluation
half. Choosing the direction on the same data the figure is reported on is what
turns an anti-correlated signal into a spurious `max(a, 1-a)` success, so the
split is not optional.

The split is repeated over several seeds. Seeds here resample the tune/eval
partition, not model training — the deployed bundle is a fixed artefact — and
the run manifest says so explicitly to keep this from being confused with the
5-seed training replication of `plantdiag-edge`.

Output: `signal_metrics.json` plus a markdown table, both generated. No number
is ever typed by hand.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from reliability_gate.metrics import aurc, auroc, bootstrap_ci, risk_coverage  # noqa: E402
from reliability_gate.signals import GalleryStats, compute_all  # noqa: E402

COVERAGES = np.array([0.10, 0.25, 0.50, 0.75, 1.00])


def git_hash(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def evaluate(
    profiles: Path, seeds: list[int], k: int, split_frac: float
) -> dict:
    dists = np.load(profiles / "distances.npy")
    gal_labels = np.load(profiles / "gallery_labels.npy")
    gg = np.load(profiles / "gallery_gg.npy")
    q_labels = np.load(profiles / "query_labels.npy")
    classes = json.loads((profiles / "classes.json").read_text())

    stats = GalleryStats.from_gallery(gg, gal_labels, len(classes))
    scores, pred = compute_all(dists, gal_labels, stats, k=k)
    correct = (pred == q_labels).astype(np.int64)
    n = correct.size

    per_signal: dict[str, dict] = {}
    for name, s in scores.items():
        finite = np.isfinite(s)
        rows = {
            "auroc": [],
            "aurc": [],
            "aurc_random": [],
            "aurc_oracle": [],
            "direction": [],
            "accuracy_at_coverage": [],
            "n_eval": [],
            "n_dropped_nonfinite": int((~finite).sum()),
        }
        for seed in seeds:
            rng = np.random.default_rng(seed)
            idx = rng.permutation(n)
            cut = int(round(split_frac * n))
            tune, ev = idx[:cut], idx[cut:]
            tune = tune[finite[tune]]
            ev = ev[finite[ev]]
            if tune.size == 0 or ev.size == 0:
                continue

            # Direction chosen on the tuning half only.
            a_tune = auroc(s[tune], correct[tune])
            direction = 1.0 if (np.isnan(a_tune) or a_tune >= 0.5) else -1.0
            se = direction * s[ev]
            ce = correct[ev]

            rows["direction"].append(direction)
            rows["auroc"].append(auroc(se, ce))
            rows["aurc"].append(aurc(se, ce))
            rows["aurc_random"].append(aurc(np.zeros_like(se), ce))
            rows["aurc_oracle"].append(aurc(ce.astype(float), ce))
            rows["accuracy_at_coverage"].append(risk_coverage(se, ce, COVERAGES).tolist())
            rows["n_eval"].append(int(ev.size))

        def agg(key: str) -> dict:
            v = np.asarray(rows[key], dtype=float)
            lo, hi = bootstrap_ci(v) if v.size > 1 else (float(v[0]), float(v[0]))
            return {
                "mean": float(v.mean()),
                "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                "ci95": [lo, hi],
                "per_seed": v.tolist(),
            }

        cov = np.asarray(rows["accuracy_at_coverage"], dtype=float)
        per_signal[name] = {
            "auroc": agg("auroc"),
            "aurc": agg("aurc"),
            "aurc_random": agg("aurc_random"),
            "aurc_oracle": agg("aurc_oracle"),
            "direction_per_seed": rows["direction"],
            "direction_stable": len(set(rows["direction"])) == 1,
            "accuracy_at_coverage": {
                f"{c:.2f}": {"mean": float(cov[:, j].mean()), "std": float(cov[:, j].std(ddof=1))}
                for j, c in enumerate(COVERAGES)
            },
            "n_dropped_nonfinite": rows["n_dropped_nonfinite"],
        }

    return {
        "per_signal": per_signal,
        "setting": {
            "n_queries": int(n),
            "n_gallery": int(gal_labels.size),
            "n_classes": len(classes),
            "k": k,
            "split_frac": split_frac,
            "seeds": seeds,
            "seed_meaning": "resamples the tune/eval partition; the bundle is a fixed artefact",
            "coverages": COVERAGES.tolist(),
            "top1_accuracy_full": float(correct.mean()),
            "macro_recall_full": float(
                np.mean([
                    (pred[q_labels == c] == c).mean()
                    for c in range(len(classes))
                    if (q_labels == c).any()
                ])
            ),
        },
        "gallery_stats": {
            "bytes": int(stats.nbytes),
            "classes_without_statistics": int(np.isnan(stats.mu).sum()),
            "mu_min": float(np.nanmin(stats.mu)),
            "mu_max": float(np.nanmax(stats.mu)),
            "mu_spread_ratio": float(np.nanmax(stats.mu) / max(np.nanmin(stats.mu), 1e-12)),
        },
    }


def markdown_table(res: dict) -> str:
    lines = [
        "| Сигнал | AUROC (mean ± std) | Направление устойчиво | AURC | AURC случайного | Точность c=0,25 | Точность c=1,00 |",
        "|---|---:|:---:|---:|---:|---:|---:|",
    ]
    for name, r in sorted(
        res["per_signal"].items(), key=lambda kv: -kv[1]["auroc"]["mean"]
    ):
        a = r["auroc"]
        lines.append(
            f"| `{name}` | {a['mean']:.3f} ± {a['std']:.3f} | "
            f"{'да' if r['direction_stable'] else '**нет**'} | "
            f"{r['aurc']['mean']:.3f} | {r['aurc_random']['mean']:.3f} | "
            f"{r['accuracy_at_coverage']['0.25']['mean']:.3f} ± {r['accuracy_at_coverage']['0.25']['std']:.3f} | "
            f"{r['accuracy_at_coverage']['1.00']['mean']:.3f} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("profiles", type=Path, help="directory produced by extract_query_profiles.py")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 42])
    ap.add_argument("--split-frac", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or args.profiles
    res = evaluate(args.profiles, args.seeds, args.k, args.split_frac)
    res["meta"] = {
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": {"edge-reliability-gate": git_hash(PROJECT)},
        "profiles_run_meta": json.loads((args.profiles / "run_meta.json").read_text()),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    (out / "signal_metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    table = markdown_table(res)
    (out / "signal_table.md").write_text(table + "\n")

    s = res["setting"]
    print(f"queries {s['n_queries']}  gallery {s['n_gallery']}  classes {s['n_classes']}  k={s['k']}")
    print(f"top-1 accuracy {s['top1_accuracy_full']:.4f}  macro-recall {s['macro_recall_full']:.4f}")
    g = res["gallery_stats"]
    print(
        f"gallery statistics: {g['bytes']} B, intra-class mu spans "
        f"{g['mu_min']:.1f}..{g['mu_max']:.1f} ({g['mu_spread_ratio']:.2f}x)"
    )
    print()
    print(table)
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
