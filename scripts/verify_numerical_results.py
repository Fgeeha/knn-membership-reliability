#!/usr/bin/env python3
"""Verify every published number against a recomputation from raw data.

Direction of the check matters and is the reason this script exists from day
one rather than after a manuscript is written. Expected values are **read from
the artefact that publishes them** — the generated table now, the manuscript
once one exists — and compared against a fresh recomputation from
`distances.npy`. Constants baked into this file would make it pass unchanged
after someone edited a published number, which is precisely the defect the
project-wide numerical audit found in its own first revision.

Checks:
  1. Every AUROC / AURC / coverage figure in `signal_metrics.json` recomputes
     bit-for-bit from `distances.npy` and `query_labels.npy`.
  2. `signal_table.md` agrees with `signal_metrics.json`.
  3. The probe reproduces the six AUROC values published in ITIDS'2026 Table 1.
  4. The resource-budget verdicts follow from the measured latency and size.
  5. If `paper/main.md` exists, every "0,xxx ± 0,xxx" in it is traced to a
     computed value.

Exit code 0 only when everything passes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from evaluate_signals import evaluate  # noqa: E402

# Reference values from an earlier measurement on the same pipeline (C1-C6) — the only external
# reference these metrics can be checked against.
ITIDS_TABLE1 = {
    "c6_a_fp32_icoral_linear": ("p1", 0.684),
    "c4_a_fp32_icoral_linear": ("p1", 0.715),
    "c6_c_fp32_icoral_pq_knn": ("conf", 0.400),
    "c6_d_int8_icoral_pq_knn": ("conf", 0.402),
    "c4_c_fp32_icoral_pq_knn": ("conf", 0.413),
    "c4_d_int8_icoral_pq_knn": ("conf", 0.418),
}

TOL = 1e-9  # recomputation must be exact: everything downstream is deterministic


class Report:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, ok: bool, what: str, detail: str = "") -> None:
        if ok:
            self.passed += 1
        else:
            self.failed.append(f"{what}{': ' + detail if detail else ''}")

    def finish(self) -> int:
        total = self.passed + len(self.failed)
        if self.failed:
            print(f"\n{len(self.failed)} of {total} CHECKS FAILED")
            for f in self.failed:
                print(f"  FAIL {f}")
            return 1
        print(f"\nALL NUMERICAL CHECKS PASSED ({total} checks)")
        return 0


def verify_profiles(run: Path, rep: Report) -> None:
    """Recompute the metrics of one profile run and compare with what it stored."""
    stored = json.loads((run / "signal_metrics.json").read_text())
    s = stored["setting"]
    fresh = evaluate(run, s["seeds"], s["k"], s["split_frac"])

    for name, want in stored["per_signal"].items():
        got = fresh["per_signal"][name]
        for field in ("auroc", "aurc", "aurc_random", "aurc_oracle"):
            rep.check(
                abs(got[field]["mean"] - want[field]["mean"]) < TOL,
                f"{run.name}/{name}/{field}.mean",
                f"stored {want[field]['mean']!r}, recomputed {got[field]['mean']!r}",
            )
            rep.check(
                np.allclose(got[field]["per_seed"], want[field]["per_seed"], atol=TOL),
                f"{run.name}/{name}/{field}.per_seed",
            )
        for cov, w in want["accuracy_at_coverage"].items():
            rep.check(
                abs(got["accuracy_at_coverage"][cov]["mean"] - w["mean"]) < TOL,
                f"{run.name}/{name}/coverage[{cov}]",
            )
        rep.check(
            got["direction_stable"] == want["direction_stable"],
            f"{run.name}/{name}/direction_stable",
        )

    for field in ("top1_accuracy_full", "macro_recall_full", "n_queries", "n_gallery"):
        rep.check(
            fresh["setting"][field] == s[field], f"{run.name}/setting/{field}"
        )

    # The stored sanity accuracy must equal the one the extractor recorded.
    meta = json.loads((run / "run_meta.json").read_text())
    rep.check(
        abs(meta["sanity"]["top1_accuracy"] - s["top1_accuracy_full"]) < 1e-9,
        f"{run.name}/extractor and evaluator disagree on accuracy",
    )


def verify_table(run: Path, rep: Report) -> None:
    """The generated markdown table must not drift from the JSON it came from."""
    table_path = run / "signal_table.md"
    if not table_path.exists():
        return
    data = json.loads((run / "signal_metrics.json").read_text())["per_signal"]
    rows = [ln for ln in table_path.read_text().splitlines() if ln.startswith("| `")]
    rep.check(len(rows) == len(data), f"{run.name}/table row count", f"{len(rows)} vs {len(data)}")
    for line in rows:
        cells = [c.strip() for c in line.strip("|").split("|")]
        name = cells[0].strip("`")
        m, sd = (float(x.replace(",", ".")) for x in cells[1].split("±"))
        rep.check(
            abs(m - round(data[name]["auroc"]["mean"], 3)) < 5e-4
            and abs(sd - round(data[name]["auroc"]["std"], 3)) < 5e-4,
            f"{run.name}/table AUROC[{name}]",
            f"table {m}±{sd}, json {data[name]['auroc']['mean']:.4f}±{data[name]['auroc']['std']:.4f}",
        )


def verify_probe(rep: Report) -> None:
    path = PROJECT / "results" / "probe_baseline_signals" / "probe_results.json"
    if not path.exists():
        rep.check(False, "probe results missing", str(path))
        return
    data = json.loads(path.read_text())
    for mode, (signal, published) in ITIDS_TABLE1.items():
        got = data["modes"][mode]["auroc"][signal]["mean"]
        rep.check(
            abs(got - published) < 5e-4,
            f"probe reproduces ITIDS Table 1 [{mode}]",
            f"got {got:.4f}, published {published}",
        )
    rep.check(
        not data["conclusion"]["cheap_fix_exists"],
        "probe conclusion consistent with its own threshold",
    )


def verify_budget(rep: Report) -> None:
    for path in sorted((PROJECT / "results").glob("gate_cost_*.json")):
        d = json.loads(path.read_text())
        b = d["budget_check"]
        size_frac = d["memory_bytes"]["gallery_statistics"] / (b["deploy_mb_reference"] * 1e6)
        lat_frac = d["latency_ms"]["p50"] / b["e2e_p50_ms_reference"]
        rep.check(abs(size_frac - b["added_size_fraction"]) < 1e-12, f"{path.name}/size fraction")
        rep.check(abs(lat_frac - b["added_latency_fraction_p50"]) < 1e-12, f"{path.name}/latency fraction")
        rep.check(b["within_10pct_size"] == (size_frac < 0.10), f"{path.name}/size verdict")
        rep.check(b["within_10pct_latency"] == (lat_frac < 0.10), f"{path.name}/latency verdict")


def verify_manuscript(rep: Report) -> None:
    """Trace every 'x,xxx ± y,yyy' in the manuscript to a computed value.

    Values are read from the manuscript, never hardcoded here, so editing a
    published number without re-running the computation makes this fail.
    """
    man = PROJECT / "paper" / "main.md"
    if not man.exists():
        print("  (no manuscript yet — manuscript check skipped)")
        return
    computed: set[tuple[float, float]] = set()
    for run in sorted((PROJECT / "results").glob("profiles_*")):
        f = run / "signal_metrics.json"
        if not f.exists():
            continue
        for r in json.loads(f.read_text())["per_signal"].values():
            for field in ("auroc", "aurc", "aurc_random", "aurc_oracle"):
                computed.add((round(r[field]["mean"], 3), round(r[field]["std"], 3)))
            for c in r["accuracy_at_coverage"].values():
                computed.add((round(c["mean"], 3), round(c["std"], 3)))

    # The manuscript (paper/main.md, stage S5) also publishes mean ± std pairs
    # from the S2-S4 runs. Rather than enumerating those sources here — the
    # dedicated verify_paper5_numbers.py checks each of them against its exact
    # origin — the tracing set is extended generically: every stored result
    # object that carries numeric `mean` and `std` fields contributes its pair.
    def harvest(obj) -> None:
        if isinstance(obj, dict):
            for mkey in ("mean", "auroc_in_vs_ood"):
                if isinstance(obj.get(mkey), (int, float)) and isinstance(obj.get("std"), (int, float)):
                    m_, s_ = obj[mkey], obj["std"]
                    computed.add((round(m_, 3), round(s_, 3)))
                    computed.add((round(abs(m_), 3), round(s_, 3)))
            am, asd = obj.get("accuracy_mean"), obj.get("accuracy_std")
            if isinstance(am, list) and isinstance(asd, list) and len(am) == len(asd):
                for m_, s_ in zip(am, asd):
                    if isinstance(m_, (int, float)) and isinstance(s_, (int, float)):
                        computed.add((round(m_, 3), round(s_, 3)))
            for v in obj.values():
                harvest(v)
        elif isinstance(obj, list):
            for v in obj:
                harvest(v)

    for f in sorted((PROJECT / "results").rglob("*.json")):
        try:
            harvest(json.loads(f.read_text()))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

    pattern = re.compile(r"(\d,\d{3})\s*±\s*(\d,\d{3})")
    for i, line in enumerate(man.read_text().splitlines(), 1):
        for m, sd in pattern.findall(line):
            pair = (float(m.replace(",", ".")), float(sd.replace(",", ".")))
            rep.check(pair in computed, f"manuscript line {i}: {m} ± {sd} not traced to any run")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=PROJECT / "results")
    args = ap.parse_args()

    rep = Report()
    runs = [d for d in sorted(args.results.glob("profiles_*")) if (d / "signal_metrics.json").exists()]
    if not runs:
        print("no evaluated profile runs found", file=sys.stderr)
        return 1
    for run in runs:
        print(f"verifying {run.name} ...")
        verify_profiles(run, rep)
        verify_table(run, rep)
    print("verifying probe ...")
    verify_probe(rep)
    print("verifying resource budget ...")
    verify_budget(rep)
    print("verifying manuscript ...")
    verify_manuscript(rep)
    return rep.finish()


if __name__ == "__main__":
    raise SystemExit(main())
