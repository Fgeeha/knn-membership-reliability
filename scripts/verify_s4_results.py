#!/usr/bin/env python3
"""Independent numerical verification of the S4 results.

Levels:
  1  integrity — every S4 result file exists, cache hashes in the extraction
     manifest match the .npy files on disk (sampled), and the alignment hash is
     identical across every S4 result that carries one.
  2  internal consistency — every stored aggregate (mean/std/ci/flags) is
     recomputed from its own stored per-seed values; every verdict flag
     (h7_supported, h8_supported, h8_cell, complementarity passes, bridge
     `all_below_half`) is recomputed from the stored numbers and the criterion
     stated *in the same file*. Nothing here is hard-coded in this script.
  3  recomputation — a sample of bridge cells is re-run from the cached C4
     features and compared to the stored per-seed AUROCs; the AUROC
     implementation itself is checked against scikit-learn on random data.

`--negative-test` copies the result tree to a temp dir, perturbs one published
number at a time (an AUROC, a verdict flag, a cache hash) and asserts the
verifier fails on each perturbation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from reliability_gate.metrics import auroc  # noqa: E402


class Report:
    def __init__(self) -> None:
        self.n = 0
        self.failures: list[str] = []

    def check(self, ok: bool, what: str, detail: str = "") -> None:
        self.n += 1
        if not ok:
            self.failures.append(f"{what}  {detail}")

    def finish(self, quiet: bool = False) -> int:
        if self.failures:
            print(f"S4 VERIFICATION FAILED: {len(self.failures)} of {self.n} checks")
            for f in self.failures:
                print("  " + f)
            return 1
        if not quiet:
            print(f"ALL S4 NUMERICAL CHECKS PASSED ({self.n} checks)")
        return 0


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_agg(rep: Report, agg: dict, where: str) -> None:
    """Recompute mean/std/flags of an aggregate from its own per_seed list."""
    if "per_seed" not in agg or agg.get("mean") is None:
        return
    a = np.asarray(agg["per_seed"], dtype=float)
    rep.check(abs(a.mean() - agg["mean"]) < 1e-9, f"{where}: mean", f"{a.mean()} vs {agg['mean']}")
    if "std" in agg and a.size > 1:
        rep.check(abs(a.std(ddof=1) - agg["std"]) < 1e-9, f"{where}: std")
    for flag, fn in (("all_below_half", lambda x: bool((x < 0.5).all())),
                     ("all_above_half", lambda x: bool((x > 0.5).all()))):
        if flag in agg:
            rep.check(agg[flag] == fn(a), f"{where}: {flag}")


def verify(results: Path, sample: int, rep: Report) -> None:
    # ── level 1: files and hashes ────────────────────────────────────────────
    files = {
        "beans": results / "s4_beans_feasibility.json",
        "membership": results / "s4_membership" / "membership_threshold_neg_d1.json",
        "boot_c6_f32": results / "s4_paired_bootstrap" / "paired_bootstrap_c6_onnx_fp32_f32.json",
        "boot_c6_pq": results / "s4_paired_bootstrap" / "paired_bootstrap_c6_onnx_fp32_pq.json",
        "boot_c4": results / "s4_paired_bootstrap" / "paired_bootstrap_c4_onnx_fp32_f32.json",
        "bridge_f32": results / "s4_bridge_c4" / "bridge_onnx_fp32_f32.json",
        "bridge_pq": results / "s4_bridge_c4" / "bridge_onnx_fp32_pq.json",
        "bridge_int8": results / "s4_bridge_c4" / "bridge_onnx_int8_pq.json",
        "extract": results / "s4_features_c4" / "s4_extract_meta.json",
        "two_pw": results / "s4_two_loop_c4" / "two_loop_plantwild_onnx_fp32.json",
        "two_pv": results / "s4_two_loop_c4" / "two_loop_plantvillage_onnx_fp32.json",
    }
    data = {}
    for name, p in files.items():
        rep.check(p.exists(), f"file missing: {p.name}")
        if p.exists():
            data[name] = json.loads(p.read_text())
    if len(data) < len(files):
        return

    meta = data["extract"]
    c4dir = results / "s4_features_c4"
    hashed = sorted(meta["caches"])
    step = max(1, len(hashed) // 6)
    for name in hashed[::step]:
        p = c4dir / name
        rep.check(p.exists(), f"cache missing: {name}")
        if p.exists():
            rep.check(sha256_file(p) == meta["caches"][name]["sha256"],
                      f"cache sha256 mismatch: {name}")

    align_hashes = {data[k]["meta"]["alignment"]["sha256"]
                    for k in ("bridge_f32", "bridge_pq", "bridge_int8", "two_pw", "two_pv")}
    rep.check(len(align_hashes) == 1, "alignment hash differs across S4 results", str(align_hashes))

    # ── level 2: internal consistency and verdict flags ─────────────────────
    rep.check(data["beans"]["conclusion"].startswith("no beans crop"), "beans conclusion text")
    rep.check(all(r["verdict"] == "OOD_UNSEEN_CROP" and not r["matches"]
                  for r in data["beans"]["rows"]),
              "beans rows must all be OOD_UNSEEN_CROP with no matches")

    for key in ("bridge_f32", "bridge_pq", "bridge_int8"):
        for r in data[key]["rows"]:
            where = f"{key} {r['query_domain']}/{r['gallery_domain']}@{r['per_class']}"
            for sig in ("neg_d1_undirected", "class_gap_undirected", "z_class_undirected", "accuracy"):
                check_agg(rep, r[sig], f"{where} {sig}")

    for key in ("two_pw", "two_pv"):
        d = data[key]
        for sig, agg in d["correctness_auroc"].items():
            check_agg(rep, agg, f"{key} correctness {sig}")
        for tier, sigs in d["membership_auroc"].items():
            for sig, agg in sigs.items():
                check_agg(rep, agg, f"{key} membership {tier} {sig}")
        ct = d["complementarity_test"]
        l1, l2 = ct["loop1_signal"], ct["loop2_signal"]
        adv_c = d["correctness_auroc"][l2]["mean"] - d["correctness_auroc"][l1]["mean"]
        adv_m = min(d["membership_auroc"][t][l1]["mean"] - d["membership_auroc"][t][l2]["mean"]
                    for t in d["membership_auroc"])
        rep.check(abs(adv_c - ct["loop2_advantage_on_correctness"]) < 1e-9, f"{key}: corr advantage")
        rep.check(abs(adv_m - ct["loop1_advantage_on_membership_min_over_tiers"]) < 1e-9,
                  f"{key}: memb advantage")
        expected_pass = (abs(d["spearman_loop1_loop2"]["mean"]) < 0.5
                         and adv_c >= 0.10 and adv_m >= 0.10)
        rep.check(ct["passes"] == expected_pass, f"{key}: complementarity verdict")

    mem = data["membership"]
    crit = mem["criteria"]
    for cname, cfg in mem["configs"].items():
        n_ok = 0
        for r in cfg["per_seed"]:
            feas = [o for o in r["operating_curve"]
                    if o["in_space_acceptance"] >= crit["in_space_acceptance"]
                    and all(o["rejection"][t] >= crit["far_ood_rejection"]
                            for t in ("fcdd_apple", "fcdd_other"))]
            rep.check(bool(feas) == r["h7_feasible"], f"membership {cname} seed {r['seed']}: feasibility")
            for o in r["operating_curve"]:
                expected = (o["in_space_acceptance"] >= crit["in_space_acceptance"]
                            and all(o["rejection"][t] >= crit["far_ood_rejection"]
                                    for t in ("fcdd_apple", "fcdd_other")))
                if o["meets_h7"] != expected:
                    rep.check(False, f"membership {cname} seed {r['seed']} q={o['q']}: meets_h7 flag")
            n_ok += bool(feas)
        rep.check(cfg["h7_main"] == (n_ok >= 4), f"membership {cname}: h7_main")
    rep.check(mem["h7_supported"] == all(c["h7_main"] for c in mem["configs"].values()),
              "membership: h7_supported flag")

    for key in ("boot_c6_f32", "boot_c6_pq", "boot_c4"):
        d = data[key]
        for r in d["rows"]:
            below = r["expected_side"] == "below 0.5"
            n_ok = 0
            for s in r["per_seed"]:
                lo, hi = s["ci95"]
                rep.check(lo <= s["auroc"] <= hi or not np.isfinite(s["auroc"]),
                          f"{key} {r['query_domain']}@{r['per_class']} seed {s['seed']}: point outside CI")
                expected = (hi < 0.5) if below else (lo > 0.5)
                rep.check(s["ci_on_expected_side"] == expected,
                          f"{key} {r['query_domain']}@{r['per_class']} seed {s['seed']}: CI side flag")
                n_ok += expected
            rep.check(r["seeds_ci_on_expected_side"] == f"{n_ok}/{len(r['per_seed'])}",
                      f"{key} {r['query_domain']}@{r['per_class']}: seed count")
            rep.check(r["h8_cell"] == (n_ok >= 4), f"{key} {r['query_domain']}@{r['per_class']}: h8_cell")
        rep.check(d["h8_supported"] == all(r["h8_cell"] for r in d["rows"]), f"{key}: h8_supported")

    # ── level 3: metric vs sklearn, and cell recomputation from raw caches ──
    try:
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(7)
        for _ in range(20):
            s = rng.normal(size=300)
            y = (rng.random(300) < 0.4).astype(int)
            rep.check(abs(auroc(s, y) - roc_auc_score(y, s)) < 1e-9, "auroc vs sklearn")
    except ImportError:
        pass

    if sample > 0 and (c4dir / "src_onnx_fp32.npy").exists():
        from s3_bridge import run as bridge_run
        s4 = {p.stem: np.load(p) for p in c4dir.glob("*.npy")}
        align = json.loads((PROJECT / "label_alignment.json").read_text())
        spec = json.loads((PROJECT.parent / "plantdiag-edge" / "configs" / "labels_clean.json").read_text())
        l2i = {l: i for i, l in enumerate(spec["target_labels"])}
        classes = np.array(sorted(l2i[v] for v in align["conservative_mapping"].values()))
        pd_lbl = s4["tgt_val_labels"]
        pd_mask = np.isin(pd_lbl, classes)
        queries = {
            "plantdoc": (s4["tgt_val_onnx_fp32"][pd_mask].astype(np.float32), pd_lbl[pd_mask]),
            "plantwild": (s4["closed_onnx_fp32"][s4["closed_query_idx"]].astype(np.float32),
                          s4["closed_labels"][s4["closed_query_idx"]]),
        }
        pools = {
            "own(matched)": {
                "plantdoc": (s4["tgt_tr_onnx_fp32"].astype(np.float32), s4["tgt_tr_labels"]),
                "plantwild": (s4["closed_onnx_fp32"][s4["closed_train_idx"]].astype(np.float32),
                              s4["closed_labels"][s4["closed_train_idx"]]),
            },
        }
        pv = (s4["src_onnx_fp32"].astype(np.float32), s4["src_labels"])
        rows = data["bridge_f32"]["rows"]
        rng = np.random.default_rng(11)
        pick = rng.choice(len(rows), size=min(sample, len(rows)), replace=False)
        seeds = data["bridge_f32"]["meta"]["seeds"]
        for i in pick:
            r = rows[i]
            qf, ql = queries[r["query_domain"]]
            pool = pools["own(matched)"][r["query_domain"]] if r["gallery_domain"] == "own(matched)" else pv
            for si, seed in enumerate(seeds[:2]):
                got = bridge_run(qf, ql, pool[0], pool[1], classes, r["per_class"], seed, compress=False)
                stored = r["neg_d1_undirected"]["per_seed"][si]
                rep.check(abs(got["neg_d1_undirected"] - stored) < 1e-9,
                          f"recompute bridge cell {r['query_domain']}@{r['per_class']} seed {seed}",
                          f"{got['neg_d1_undirected']} vs {stored}")


def negative_test(results: Path) -> int:
    """Perturb one published value at a time; the verifier must fail on each."""
    perturbations = []

    def p1(root: Path) -> None:  # change a stored AUROC per-seed value
        f = root / "s4_bridge_c4" / "bridge_onnx_fp32_f32.json"
        d = json.loads(f.read_text())
        d["rows"][1]["neg_d1_undirected"]["per_seed"][0] += 0.05
        f.write_text(json.dumps(d))

    def p2(root: Path) -> None:  # flip a verdict flag
        f = root / "s4_paired_bootstrap" / "paired_bootstrap_c6_onnx_fp32_f32.json"
        d = json.loads(f.read_text())
        d["h8_supported"] = True
        f.write_text(json.dumps(d))

    def p3(root: Path) -> None:  # corrupt a cache hash in the manifest
        f = root / "s4_features_c4" / "s4_extract_meta.json"
        d = json.loads(f.read_text())
        k = sorted(d["caches"])[0]
        d["caches"][k]["sha256"] = "0" * 64
        f.write_text(json.dumps(d))

    def p4(root: Path) -> None:  # flip the membership verdict
        f = root / "s4_membership" / "membership_threshold_neg_d1.json"
        d = json.loads(f.read_text())
        d["h7_supported"] = True
        f.write_text(json.dumps(d))

    perturbations = [("bridge per-seed AUROC +0.05", p1), ("h8_supported flipped", p2),
                     ("cache sha256 corrupted", p3), ("h7_supported flipped", p4)]

    def make_copy(td: Path) -> Path:
        root = td / "results"
        shutil.copytree(results, root,
                        ignore=shutil.ignore_patterns("*.npy", "s2_*", "s3_*", "probe_*",
                                                      "profiles_*", "*.log"))
        # labels/idx npy are small; the big feature caches are not copied, so the
        # manifest is reduced to the files that exist — a pristine copy must PASS,
        # otherwise every perturbation would be "detected" for the wrong reason.
        for k in ("closed_labels", "closed_query_idx", "closed_train_idx"):
            src = results / "s4_features_c4" / f"{k}.npy"
            if src.exists():
                shutil.copy2(src, root / "s4_features_c4" / f"{k}.npy")
        mf = root / "s4_features_c4" / "s4_extract_meta.json"
        d = json.loads(mf.read_text())
        d["caches"] = {k: v for k, v in d["caches"].items()
                       if (root / "s4_features_c4" / k).exists()}
        mf.write_text(json.dumps(d))
        return root

    with tempfile.TemporaryDirectory() as td:
        rep = Report()
        verify(make_copy(Path(td)), sample=0, rep=rep)
        if rep.finish(quiet=True) != 0:
            print("NEGATIVE TEST INVALID: the unperturbed reduced copy does not pass")
            for f in rep.failures[:5]:
                print("  " + f)
            return 1
        print("baseline (unperturbed reduced copy): passes")

    failures = 0
    for name, fn in perturbations:
        with tempfile.TemporaryDirectory() as td:
            root = make_copy(Path(td))
            fn(root)
            rep = Report()
            verify(root, sample=0, rep=rep)
            rc = rep.finish(quiet=True)
            status = "detected" if rc != 0 else "MISSED"
            print(f"negative test [{name}]: {status}")
            if rc == 0:
                failures += 1
    if failures:
        print(f"NEGATIVE TEST FAILED: {failures} perturbation(s) went undetected")
        return 1
    print("all perturbations detected")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=PROJECT / "results")
    ap.add_argument("--sample", type=int, default=4, help="bridge cells to recompute end-to-end")
    ap.add_argument("--negative-test", action="store_true")
    args = ap.parse_args()
    if args.negative_test:
        return negative_test(args.results)
    rep = Report()
    verify(args.results, args.sample, rep)
    return rep.finish()


if __name__ == "__main__":
    raise SystemExit(main())
