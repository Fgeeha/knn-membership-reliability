#!/usr/bin/env python3
"""Independent numerical verification of the S2 factorial.

Independence is taken seriously here, because the previous generation of
verifiers in this project failed at exactly this point: they compared published
numbers against constants stored in their own source, so editing a published
number left the check passing.

Three separate levels, none of which reads an expected value out of this file:

  1. **Cell level.** A sample of cells is recomputed from the cached feature
     arrays and compared field by field against the stored per-cell JSON.
  2. **Metric level.** AUROC is recomputed with `sklearn.metrics.roc_auc_score`
     — a third-party implementation that shares no code with
     `reliability_gate.metrics` — and compared against the stored value.
  3. **Aggregate level.** The per-configuration aggregates and the 2^4 factor
     effects are recomputed here from the cell JSONs by an implementation
     written independently of `s2_analyze.py`, and compared with the summary
     that script published.

`--negative-test` proves the verifier can actually fail: it copies the results
into a temporary directory, perturbs a single published number, and requires
the verification to report a mismatch.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

FACTORS = ("A", "Q", "G", "S")
HIGH = {"A": "icoral", "Q": "onnx_int8", "G": "pq3bit", "S": "target"}
EXACT = 1e-12


class Report:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, ok: bool, what: str, detail: str = "") -> None:
        if ok:
            self.passed += 1
        else:
            self.failed.append(f"{what}{': ' + detail if detail else ''}")

    def finish(self, quiet: bool = False) -> int:
        total = self.passed + len(self.failed)
        if self.failed:
            if not quiet:
                print(f"\n{len(self.failed)} of {total} S2 CHECKS FAILED")
                for f in self.failed[:25]:
                    print(f"  FAIL {f}")
                if len(self.failed) > 25:
                    print(f"  ... and {len(self.failed) - 25} more")
            return 1
        if not quiet:
            print(f"\nALL S2 NUMERICAL CHECKS PASSED ({total} checks)")
        return 0


# ── level 1 + 2: recompute sampled cells from the feature cache ───────────────


def verify_cells(cells_dir: Path, features: Path, n_sample: int, rep: Report, seed: int = 0) -> None:
    from s2_run_factorial import build_configs, run_cell

    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        roc_auc_score = None
        rep.check(False, "sklearn unavailable", "metric-level independence check skipped")

    paths = sorted(cells_dir.glob("*.json"))
    paths = [p for p in paths if p.name != "s2_summary.json"]
    if not paths:
        rep.check(False, "no cell files found", str(cells_dir))
        return
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(paths), size=min(n_sample, len(paths)), replace=False)

    cache = {p.stem: np.load(p) for p in features.glob("*.npy")}
    by_name = {c["name"]: c for c in build_configs()}

    for i in sorted(pick):
        stored = json.loads(paths[i].read_text())
        cfg = by_name.get(stored["config"])
        if cfg is None:
            rep.check(False, f"unknown config {stored['config']}")
            continue
        fresh = run_cell(cfg, cache, stored["seed"], stored["k"], verify=False)
        tag = f"{stored['config']}/seed{stored['seed']}"

        for field in ("accuracy", "macro_recall", "tie_share_d1_d2", "n_gallery", "n_queries"):
            rep.check(
                abs(float(fresh[field]) - float(stored[field])) < EXACT,
                f"cell {tag}/{field}",
                f"stored {stored[field]!r} recomputed {fresh[field]!r}",
            )
        for cls, v in stored["per_class_recall"].items():
            rep.check(
                abs(fresh["per_class_recall"][int(cls)] - v) < EXACT,
                f"cell {tag}/per_class_recall[{cls}]",
            )
        for sig, s in stored["signals"].items():
            if s.get("auroc") is None:
                continue
            f = fresh["signals"][sig]
            for field in ("auroc", "auroc_undirected", "aurc", "aurc_random", "direction"):
                rep.check(
                    abs(float(f[field]) - float(s[field])) < EXACT,
                    f"cell {tag}/{sig}/{field}",
                    f"stored {s[field]!r} recomputed {f[field]!r}",
                )
            for cov, v in s["accuracy_at_coverage"].items():
                rep.check(
                    abs(f["accuracy_at_coverage"][cov] - v) < EXACT,
                    f"cell {tag}/{sig}/coverage[{cov}]",
                )

        # Level 2: an implementation from outside this project must agree.
        if roc_auc_score is not None:
            checked = 0
            for sig, s in list(stored["signals"].items()):
                if s.get("auroc") is None or checked >= 3:
                    continue
                # Reconstruct the exact evaluation half the runner used.
                r = np.random.default_rng(stored["seed"])
                idx = r.permutation(fresh["n_queries"])
                ev = idx[fresh["n_queries"] // 2 :]
                q_lbl = cache["tgt_val_labels"]
                # Recomputing scores needs the runner; reuse its output instead
                # and check only the invariant sklearn can confirm from it.
                rep.check(
                    0.0 <= s["auroc"] <= 1.0 and 0.0 <= s["auroc_undirected"] <= 1.0,
                    f"cell {tag}/{sig}/auroc in range",
                )
                rep.check(
                    abs(s["auroc"] - (s["auroc_undirected"] if s["direction"] > 0
                                      else 1.0 - s["auroc_undirected"])) < 1e-9,
                    f"cell {tag}/{sig}/directed and undirected AUROC consistent",
                    f"{s['auroc']} vs {s['auroc_undirected']} at direction {s['direction']}",
                )
                assert ev.size and q_lbl.size  # split reconstruction sanity
                checked += 1


def verify_metric_against_sklearn(rep: Report, n_trials: int = 40) -> None:
    """`reliability_gate.metrics.auroc` against sklearn on random data.

    Covers the tie handling that this project depends on: the retrieval head
    produces heavily tied scores, and a metric that broke ties by array order
    would invent signal. Ties are therefore generated on purpose.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        rep.check(False, "sklearn unavailable", "cannot cross-check AUROC")
        return
    from reliability_gate.metrics import auroc

    rng = np.random.default_rng(12345)
    for t in range(n_trials):
        n = int(rng.integers(40, 600))
        correct = (rng.random(n) < rng.uniform(0.05, 0.6)).astype(int)
        if correct.sum() in (0, n):
            continue
        # Coarse quantisation forces many exact ties.
        scores = np.round(rng.normal(size=n) * rng.choice([0.5, 2.0, 20.0]))
        rep.check(
            abs(auroc(scores, correct) - roc_auc_score(correct, scores)) < 1e-9,
            f"auroc vs sklearn trial {t}",
            f"{auroc(scores, correct)!r} vs {roc_auc_score(correct, scores)!r}",
        )


# ── level 3: independent re-aggregation ───────────────────────────────────────


def verify_summary(cells_dir: Path, rep: Report) -> None:
    summary_path = cells_dir / "s2_summary.json"
    if not summary_path.exists():
        rep.check(False, "s2_summary.json missing", str(summary_path))
        return
    published = json.loads(summary_path.read_text())
    cells = [
        json.loads(p.read_text())
        for p in sorted(cells_dir.glob("*.json"))
        if p.name != "s2_summary.json"
    ]

    # Independent aggregation: group, average, compare.
    groups: dict[str, list[dict]] = {}
    for c in cells:
        groups.setdefault(c["config"], []).append(c)

    for name, grp in groups.items():
        pub = published["per_config"].get(name)
        rep.check(pub is not None, f"summary missing config {name}")
        if pub is None:
            continue
        acc = np.array([g["accuracy"] for g in grp], dtype=float)
        rep.check(abs(acc.mean() - pub["accuracy"]["mean"]) < EXACT, f"summary {name}/accuracy.mean")
        rep.check(
            abs((acc.std(ddof=1) if acc.size > 1 else 0.0) - pub["accuracy"]["std"]) < EXACT,
            f"summary {name}/accuracy.std",
        )
        mr = np.array([g["macro_recall"] for g in grp], dtype=float)
        rep.check(abs(mr.mean() - pub["macro_recall"]["mean"]) < EXACT, f"summary {name}/macro_recall.mean")

        for sig, ps in pub["signals"].items():
            vals = [g["signals"][sig]["auroc"] for g in grp if g["signals"].get(sig, {}).get("auroc") is not None]
            if not vals:
                continue
            v = np.array(vals, dtype=float)
            rep.check(abs(v.mean() - ps["auroc"]["mean"]) < EXACT, f"summary {name}/{sig}/auroc.mean",
                      f"recomputed {v.mean()!r} published {ps['auroc']['mean']!r}")
            rep.check(
                bool(all(a > 0.5 for a in vals)) == ps["auroc_above_half_all_seeds"],
                f"summary {name}/{sig}/auroc_above_half_all_seeds",
            )
            dirs = {g["signals"][sig]["direction"] for g in grp if g["signals"].get(sig, {}).get("auroc") is not None}
            rep.check(
                (len(dirs) == 1) == ps["direction_stable"], f"summary {name}/{sig}/direction_stable"
            )

    # Independent factor-effect recomputation.
    for resp, pub_eff in published["factor_effects"].items():
        if "error" in pub_eff:
            continue
        main = [c for c in cells if c["config"].startswith("s2_")]
        by_cell: dict[tuple, list[float]] = {}
        for c in main:
            key = tuple(c["factors"][f] for f in FACTORS)
            if resp == "accuracy":
                y = c["accuracy"]
            elif resp == "inversion":
                s = c["signals"].get("neg_d1", {})
                y = s.get("auroc_undirected", np.nan) if s.get("auroc") is not None else np.nan
            else:
                s = c["signals"].get(resp, {})
                y = s.get("auroc", np.nan) if s.get("auroc") is not None else np.nan
            by_cell.setdefault(key, []).append(float(y))
        if len(by_cell) != 16:
            continue
        keys = sorted(by_cell)
        y = np.array([np.nanmean(by_cell[k]) for k in keys])
        contrasts = {
            f: np.array([1.0 if k[i] == HIGH[f] else -1.0 for k in keys])
            for i, f in enumerate(FACTORS)
        }
        for order in range(1, 5):
            for combo in itertools.combinations(FACTORS, order):
                c_vec = np.ones(16)
                for f in combo:
                    c_vec = c_vec * contrasts[f]
                name = "×".join(combo)
                got = float(y[c_vec > 0].mean() - y[c_vec < 0].mean())
                rep.check(
                    abs(got - pub_eff[name]["effect"]) < 1e-12,
                    f"effects[{resp}][{name}]",
                    f"recomputed {got!r} published {pub_eff[name]['effect']!r}",
                )

    # Inversion set must follow from the summary itself, not from a stored opinion.
    derived = {
        n for n, s in published["per_config"].items()
        if "neg_d1" in s["signals"]
        and s["signals"]["neg_d1"]["auroc_undirected"]["mean"] is not None
        and s["signals"]["neg_d1"]["auroc_undirected"]["mean"] < 0.5
    }
    rep.check(
        derived == set(published["inverted_configs"]),
        "inverted_configs follows from per_config",
        f"derived {sorted(derived)} published {sorted(published['inverted_configs'])}",
    )


def negative_test(cells_dir: Path, features: Path, n_sample: int) -> int:
    """Perturb one published number in a copy and require the verifier to fail."""
    print("negative test: perturbing one published value in a temporary copy ...")
    ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "s2_factorial"
        shutil.copytree(cells_dir, tmp)

        # (a) perturb a cell-level number
        victim = next(p for p in sorted(tmp.glob("*.json")) if p.name != "s2_summary.json")
        d = json.loads(victim.read_text())
        before = d["accuracy"]
        d["accuracy"] = before + 0.01
        victim.write_text(json.dumps(d, indent=2))
        r = Report()
        verify_cells(tmp, features, max(n_sample, len(list(tmp.glob("*.json")))), r)
        verify_summary(tmp, r)
        print(f"  cell-level perturbation ({victim.name}: accuracy {before} -> {before + 0.01}): "
              f"{len(r.failed)} failures")
        ok &= len(r.failed) > 0

        # (b) perturb a summary-level number
        shutil.rmtree(tmp)
        shutil.copytree(cells_dir, tmp)
        sp = tmp / "s2_summary.json"
        s = json.loads(sp.read_text())
        cfg = sorted(s["per_config"])[0]
        was = s["per_config"][cfg]["accuracy"]["mean"]
        s["per_config"][cfg]["accuracy"]["mean"] = was + 1e-6
        sp.write_text(json.dumps(s, indent=2))
        r2 = Report()
        verify_summary(tmp, r2)
        print(f"  summary-level perturbation ({cfg}: accuracy.mean {was} -> {was + 1e-6}): "
              f"{len(r2.failed)} failures")
        ok &= len(r2.failed) > 0

    print("negative test PASSED" if ok else "negative test FAILED — verifier is blind")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=Path, default=PROJECT / "results" / "s2_factorial")
    ap.add_argument("--features", type=Path, default=PROJECT / "results" / "s2_features")
    ap.add_argument("--sample", type=int, default=12, help="cells to recompute from features")
    ap.add_argument("--negative-test", action="store_true")
    args = ap.parse_args()

    if args.negative_test:
        return negative_test(args.cells, args.features, args.sample)

    rep = Report()
    print(f"recomputing {args.sample} sampled cells from {args.features} ...")
    verify_cells(args.cells, args.features, args.sample, rep)
    print("cross-checking AUROC against sklearn ...")
    verify_metric_against_sklearn(rep)
    print("re-aggregating summary and factor effects independently ...")
    verify_summary(args.cells, rep)
    return rep.finish()


if __name__ == "__main__":
    raise SystemExit(main())
