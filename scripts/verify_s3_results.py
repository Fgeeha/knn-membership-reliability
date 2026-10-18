#!/usr/bin/env python3
"""Independent numerical verification of S3.

Same discipline as `verify_s3`'s predecessor: no expected value is stored in this
file. Everything is either recomputed from the caches and the alignment, or
re-derived from the raw per-cell records by an implementation written separately
from the one that produced the summary.

Levels:
  1. **Data.** Image counts per group, class counts, split sizes and the
     no-overlap property of the gallery/query split are recomputed from the
     alignment file and the directory tree.
  2. **Alignment.** The mapping hash is recomputed from the mapping content and
     compared with the stored one; every result file must cite that same hash.
  3. **Cache.** Every cached array's sha256, shape and dtype are recomputed.
  4. **Cells.** A sample of factorial cells is recomputed end to end from the
     caches and compared field by field.
  5. **Metrics.** AUROC is cross-checked against `sklearn.metrics.roc_auc_score`.
  6. **Summary.** Per-configuration aggregates and the primary endpoint are
     recomputed here, independently of `s3_analyze.py`.

`--negative-test` perturbs, one at a time: a summary AUROC, the alignment
mapping, one raw cell result, and the sign of the primary effect. Each must make
verification fail, and the copies are discarded afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
PLANTDIAG = PROJECT.parent / "plantdiag-edge"
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PLANTDIAG))

EXACT = 1e-12
MATCHED = "plantwild"
MISMATCHED = ("plantvillage", "plantdoc")


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
                print(f"\n{len(self.failed)} of {total} S3 CHECKS FAILED")
                for f in self.failed[:25]:
                    print(f"  FAIL {f}")
                if len(self.failed) > 25:
                    print(f"  ... and {len(self.failed) - 25} more")
            return 1
        if not quiet:
            print(f"\nALL S3 NUMERICAL CHECKS PASSED ({total} checks)")
        return 0


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_alignment(align_path: Path, rep: Report) -> dict:
    a = json.loads(align_path.read_text())
    stable = json.dumps(
        {"version": a["mapping_version"], "rows": a["rows"],
         "conservative": a["conservative_mapping"]},
        sort_keys=True, ensure_ascii=False,
    )
    rep.check(
        hashlib.sha256(stable.encode()).hexdigest() == a["mapping_sha256"],
        "alignment hash matches its own content",
    )
    counts = a["counts"]
    by_action: dict[str, list] = {}
    for r in a["rows"]:
        by_action.setdefault(r["action"], []).append(r)
    rep.check(len(a["rows"]) == counts["external_classes"], "alignment/external_classes")
    rep.check(len(by_action.get("INCLUDE_CONSERVATIVE", [])) == counts["conservative_included"],
              "alignment/conservative_included")
    rep.check(
        sum(r["n_images"] for r in by_action.get("INCLUDE_CONSERVATIVE", []))
        == counts["conservative_images"], "alignment/conservative_images")
    rep.check(len(set(a["conservative_mapping"].values())) == len(a["conservative_mapping"]),
              "alignment: no canonical label claimed twice")
    for r in a["rows"]:
        if r["action"] == "INCLUDE_CONSERVATIVE":
            rep.check(r["match_type"] in ("EXACT", "NORMALIZED_EXACT"),
                      f"alignment: {r['external_label']} auto-accepted only on exact evidence",
                      r["match_type"])
    return a


def verify_data(align: dict, s3_dir: Path, plantwild: Path, rep: Report) -> None:
    meta = json.loads((s3_dir / "s3_extract_meta.json").read_text())
    rep.check(meta["alignment"]["sha256"] == align["mapping_sha256"],
              "feature cache cites the same alignment hash")
    action_to_group = {
        "INCLUDE_CONSERVATIVE": "closed", "OOD_KNOWN_CROP": "ood_known_crop",
        "OOD_UNSEEN_CROP": "ood_unseen_crop", "AUTHOR_DECISION": "author_pending",
    }
    expect: dict[str, int] = {}
    for r in align["rows"]:
        g = action_to_group[r["action"]]
        n = len(list((plantwild / r["external_label"]).glob("*.jpg")))
        rep.check(n == r["n_images"], f"image count on disk for {r['external_label']}",
                  f"disk {n} alignment {r['n_images']}")
        expect[g] = expect.get(g, 0) + n
    for g, n in expect.items():
        rep.check(meta["groups"][g]["n_images"] == n, f"group size {g}",
                  f"manifest {meta['groups'][g]['n_images']} recomputed {n}")

    tr = np.load(s3_dir / "closed_train_idx.npy")
    qi = np.load(s3_dir / "closed_query_idx.npy")
    lab = np.load(s3_dir / "closed_labels.npy")
    rep.check(len(set(tr.tolist()) & set(qi.tolist())) == 0,
              "LEAKAGE: gallery pool and query set are disjoint")
    rep.check(tr.size + qi.size == lab.size, "split covers the closed set exactly")
    rep.check(qi.size == meta["groups"]["closed"]["n_query"], "query count matches manifest")

    for name, rec in meta["caches"].items():
        p = s3_dir / name
        if not p.exists():
            rep.check(False, f"cache file missing: {name}")
            continue
        rep.check(sha256_file(p) == rec["sha256"], f"cache sha256 {name}")
        a = np.load(p, mmap_mode="r")
        rep.check(list(a.shape) == rec["shape"] and str(a.dtype) == rec["dtype"],
                  f"cache shape/dtype {name}")


def verify_cells(cells_dir: Path, s2_dir: Path, s3_dir: Path, align: dict,
                 n_sample: int, rep: Report, seed: int = 0) -> None:
    from s3_run_factorial import build_cell, build_configs, evaluate_cell, load_pools

    paths = [p for p in sorted(cells_dir.glob("*.json")) if not p.name.startswith("s3_summary")]
    if not paths:
        rep.check(False, "no S3 cell files found")
        return
    pools = load_pools(s2_dir, s3_dir)
    by_name = {c["name"]: c for c in build_configs()}
    rng = np.random.default_rng(seed)
    for i in sorted(rng.choice(len(paths), size=min(n_sample, len(paths)), replace=False)):
        stored = json.loads(paths[i].read_text())
        cfg = by_name.get(stored["config"])
        if cfg is None:
            rep.check(False, f"unknown config {stored['config']}")
            continue
        rep.check(stored["meta"]["alignment"]["sha256"] == align["mapping_sha256"],
                  f"{paths[i].name} cites the verified alignment")
        built = build_cell(cfg, pools, stored["seed"], stored["per_class"], stored["k_rule"])
        fresh = evaluate_cell(built, stored["seed"])
        tag = f"{stored['config']}/seed{stored['seed']}/{stored['k_rule']}"
        for f in ("accuracy", "macro_recall", "n_gallery", "n_queries", "k", "tie_share_d1_d2"):
            rep.check(abs(float(fresh[f]) - float(stored[f])) < EXACT, f"cell {tag}/{f}",
                      f"stored {stored[f]!r} recomputed {fresh[f]!r}")
        for sig, s in stored["signals"].items():
            if s.get("auroc") is None:
                continue
            g = fresh["signals"][sig]
            for f in ("auroc", "auroc_undirected", "auprc", "aurc", "direction"):
                rep.check(abs(float(g[f]) - float(s[f])) < EXACT, f"cell {tag}/{sig}/{f}",
                          f"stored {s[f]!r} recomputed {g[f]!r}")
            rep.check(
                abs(s["auroc"] - (s["auroc_undirected"] if s["direction"] > 0
                                  else 1.0 - s["auroc_undirected"])) < 1e-9,
                f"cell {tag}/{sig}: directed and undirected AUROC consistent")


def verify_metric_against_sklearn(rep: Report, n_trials: int = 30) -> None:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        rep.check(False, "sklearn unavailable")
        return
    from reliability_gate.metrics import auroc

    rng = np.random.default_rng(999)
    for t in range(n_trials):
        n = int(rng.integers(40, 500))
        y = (rng.random(n) < rng.uniform(0.1, 0.7)).astype(int)
        if y.sum() in (0, n):
            continue
        s = np.round(rng.normal(size=n) * rng.choice([1.0, 10.0]))
        rep.check(abs(auroc(s, y) - roc_auc_score(y, s)) < 1e-9, f"auroc vs sklearn {t}")


def verify_summary(cells_dir: Path, rep: Report) -> None:
    for sp in sorted(cells_dir.glob("s3_summary_*.json")):
        pub = json.loads(sp.read_text())
        arm, krule = pub["arm"], pub["k_rule"]
        cells = [json.loads(p.read_text()) for p in sorted(cells_dir.glob("*.json"))
                 if not p.name.startswith("s3_summary")]
        cells = [c for c in cells if c["factors"]["A"] == arm and c["k_rule"] == krule]
        rep.check(len(cells) == pub["meta"]["n_cells"], f"{sp.name}/n_cells")

        groups: dict[str, list] = {}
        for c in cells:
            groups.setdefault(c["config"], []).append(c)
        for name, grp in groups.items():
            p = pub["per_config"].get(name)
            rep.check(p is not None, f"{sp.name}: missing config {name}")
            if p is None:
                continue
            acc = np.array([g["accuracy"] for g in grp], float)
            rep.check(abs(acc.mean() - p["accuracy"]["mean"]) < EXACT, f"{sp.name}/{name}/accuracy")
            for sig, ps in p["signals"].items():
                v = np.array([g["signals"][sig]["auroc"] for g in grp
                              if g["signals"].get(sig, {}).get("auroc") is not None], float)
                if v.size:
                    rep.check(abs(v.mean() - ps["auroc"]["mean"]) < EXACT,
                              f"{sp.name}/{name}/{sig}/auroc.mean")

        # Primary endpoint, recomputed independently and paired by seed.
        by_seed: dict[int, dict[str, list[float]]] = {}
        for c in cells:
            lvl = c["factors"]["S"]
            side = "a" if lvl == MATCHED else "b"
            v = c["signals"]["neg_d1"]["auroc_undirected"]
            by_seed.setdefault(c["seed"], {"a": [], "b": []})[side].append(v)
        diffs = [float(np.nanmean(d["a"]) - np.nanmean(d["b"]))
                 for _, d in sorted(by_seed.items()) if d["a"] and d["b"]]
        got = float(np.mean(diffs))
        want = pub["primary_endpoint"]["matched_minus_mismatched_pooled"]["mean"]
        rep.check(abs(got - want) < 1e-12, f"{sp.name}/primary endpoint",
                  f"recomputed {got!r} published {want!r}")
        rep.check(
            np.allclose(diffs, pub["primary_endpoint"]["matched_minus_mismatched_pooled"]["per_seed"],
                        atol=1e-12),
            f"{sp.name}/primary endpoint per-seed")

        # Inversion verdicts must follow from the numbers, not be asserted.
        for lvl, b in pub["by_gallery_domain"].items():
            for sig, d in b["signals"].items():
                if d["auroc_undirected"]["mean"] is None:
                    continue
                rep.check((d["auroc_undirected"]["mean"] < 0.5) == d["inverted"],
                          f"{sp.name}/{lvl}/{sig}/inverted follows from AUROC")


def negative_test(cells_dir: Path, s2_dir: Path, s3_dir: Path, align_path: Path,
                  plantwild: Path, n_sample: int) -> int:
    print("negative test: four perturbations, each must make verification fail\n")
    ok = True
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)

        # 1. summary AUROC
        tmp = base / "cells1"
        shutil.copytree(cells_dir, tmp)
        sp = next(tmp.glob("s3_summary_*.json"))
        d = json.loads(sp.read_text())
        cfg = sorted(d["per_config"])[0]
        was = d["per_config"][cfg]["accuracy"]["mean"]
        d["per_config"][cfg]["accuracy"]["mean"] = was + 1e-6
        sp.write_text(json.dumps(d, indent=2))
        r = Report(); verify_summary(tmp, r)
        print(f"  1. summary accuracy {was} -> {was + 1e-6}: {len(r.failed)} failures")
        ok &= len(r.failed) > 0

        # 2. alignment mapping
        ap = base / "align.json"
        a = json.loads(align_path.read_text())
        a["conservative_mapping"]["apple scab"] = "Tomato_leaf"
        ap.write_text(json.dumps(a, indent=2))
        r = Report(); verify_alignment(ap, r)
        print(f"  2. alignment mapping altered: {len(r.failed)} failures")
        ok &= len(r.failed) > 0

        # 3. one raw cell result
        tmp3 = base / "cells3"
        shutil.copytree(cells_dir, tmp3)
        victim = next(p for p in sorted(tmp3.glob("*.json")) if not p.name.startswith("s3_summary"))
        c = json.loads(victim.read_text())
        wasa = c["signals"]["class_gap"]["auroc"]
        c["signals"]["class_gap"]["auroc"] = wasa + 0.05
        victim.write_text(json.dumps(c, indent=2))
        r = Report()
        verify_cells(tmp3, s2_dir, s3_dir, json.loads(align_path.read_text()),
                     max(n_sample, 40), r)
        verify_summary(tmp3, r)
        print(f"  3. raw cell class_gap {wasa:.4f} -> {wasa + 0.05:.4f}: {len(r.failed)} failures")
        ok &= len(r.failed) > 0

        # 4. sign of the primary effect
        tmp4 = base / "cells4"
        shutil.copytree(cells_dir, tmp4)
        sp4 = next(tmp4.glob("s3_summary_*.json"))
        d4 = json.loads(sp4.read_text())
        pe = d4["primary_endpoint"]["matched_minus_mismatched_pooled"]
        before = pe["mean"]
        pe["mean"] = -before
        pe["per_seed"] = [-x for x in pe["per_seed"]]
        sp4.write_text(json.dumps(d4, indent=2))
        r = Report(); verify_summary(tmp4, r)
        print(f"  4. primary effect sign {before:+.4f} -> {-before:+.4f}: {len(r.failed)} failures")
        ok &= len(r.failed) > 0

    print("\nnegative test PASSED" if ok else "\nnegative test FAILED — verifier is blind")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=Path, default=PROJECT / "results" / "s3_factorial")
    ap.add_argument("--s2-features", type=Path, default=PROJECT / "results" / "s2_features")
    ap.add_argument("--s3-features", type=Path, default=PROJECT / "results" / "s3_features")
    ap.add_argument("--alignment", type=Path, default=PROJECT / "label_alignment.json")
    ap.add_argument("--plantwild", type=Path,
                    default=Path("data/plantwild_v2"))
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--negative-test", action="store_true")
    args = ap.parse_args()

    if args.negative_test:
        return negative_test(args.cells, args.s2_features, args.s3_features,
                             args.alignment, args.plantwild, args.sample)

    rep = Report()
    print("verifying label alignment ...")
    align = verify_alignment(args.alignment, rep)
    print("verifying dataset counts, splits and cache hashes ...")
    verify_data(align, args.s3_features, args.plantwild, rep)
    print(f"recomputing {args.sample} sampled cells from the caches ...")
    verify_cells(args.cells, args.s2_features, args.s3_features, align, args.sample, rep)
    print("cross-checking AUROC against sklearn ...")
    verify_metric_against_sklearn(rep)
    print("re-deriving summaries and the primary endpoint independently ...")
    verify_summary(args.cells, rep)
    return rep.finish()


if __name__ == "__main__":
    raise SystemExit(main())
