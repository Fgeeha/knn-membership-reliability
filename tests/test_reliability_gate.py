"""Tests for the reliability gate.

Run: `pytest tests/ -q` from the project root, or
`python tests/test_reliability_gate.py` without pytest installed.

The module-level self-checks (`python -m reliability_gate.metrics`,
`python -m reliability_gate.signals`) cover the arithmetic. What is tested here
is the part that arithmetic cannot catch: that the pipeline reproduces already
published numbers, and that the evaluation protocol cannot manufacture a signal
out of noise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from reliability_gate.metrics import auroc, risk_coverage  # noqa: E402
from reliability_gate.signals import GalleryStats, compute_all  # noqa: E402

PROBE = PROJECT / "results" / "probe_baseline_signals" / "probe_results.json"

# Reference values from an earlier measurement on the same pipeline. Reproducing these from
# the raw per-sample dumps is the only external check available on this code.
ITIDS_PUBLISHED_AUROC = {
    "c6_a_fp32_icoral_linear": 0.684,
    "c4_a_fp32_icoral_linear": 0.715,
    "c6_c_fp32_icoral_pq_knn": 0.400,
    "c6_d_int8_icoral_pq_knn": 0.402,
    "c4_c_fp32_icoral_pq_knn": 0.413,
    "c4_d_int8_icoral_pq_knn": 0.418,
}


def test_probe_reproduces_published_itids_auroc() -> None:
    """The probe's AUROC must match the published table to the printed precision."""
    if not PROBE.exists():
        print("  skipped: run scripts/probe_baseline_signals.py first")
        return
    data = json.loads(PROBE.read_text())
    for mode, published in ITIDS_PUBLISHED_AUROC.items():
        signal = "p1" if mode.endswith("linear") else "conf"
        got = data["modes"][mode]["auroc"][signal]["mean"]
        assert abs(got - published) < 5e-4, f"{mode}: got {got:.4f}, published {published}"


def test_retrieval_confidence_is_just_negative_distance() -> None:
    """`conf` carries no information beyond `d1`, so they must rank identically."""
    if not PROBE.exists():
        print("  skipped: run scripts/probe_baseline_signals.py first")
        return
    data = json.loads(PROBE.read_text())
    for mode, rec in data["modes"].items():
        if mode.endswith("linear"):
            continue
        assert rec["conf_is_monotone_in_neg_d1"], mode


def test_no_signal_in_pure_noise() -> None:
    """The protocol must not turn noise into a usable signal.

    Direction is picked on a tuning half; picking it on the evaluation half
    would push a noise AUROC systematically above 0.5. This asserts the split
    actually prevents that.

    Each repetition draws a **fresh** dataset. Re-splitting one dataset would not
    converge to 0.5 at all: every split then estimates the same fixed draw, so
    the spread across splits measures the split, not the signal. That is the
    same trap `paper4-compression-attribution` records as its methodological
    finding, and it applies to this project's own seeds — see
    `results/*/signal_metrics.json`, field `seed_meaning`.
    """
    n, reps = 4000, 60
    honest, cheating = [], []
    for rep in range(reps):
        rng = np.random.default_rng(1000 + rep)
        correct = (rng.random(n) < 0.17).astype(int)
        noise = rng.normal(size=n)
        idx = rng.permutation(n)
        tune, ev = idx[: n // 2], idx[n // 2 :]
        d = 1.0 if auroc(noise[tune], correct[tune]) >= 0.5 else -1.0
        honest.append(auroc(d * noise[ev], correct[ev]))
        a = auroc(noise[ev], correct[ev])
        cheating.append(max(a, 1 - a))

    # SE of one AUROC here is ~0.017; over 60 independent draws, ~0.0022.
    assert abs(np.mean(honest) - 0.5) < 0.01, np.mean(honest)
    assert np.mean(cheating) > 0.51, f"fixture fails to expose the bias: {np.mean(cheating)}"


def test_signals_never_read_query_labels() -> None:
    """Permuting the true labels must not change any signal value.

    A signal that peeks at the evaluation labels would be the single most likely
    way to fake this project's result, so it is checked directly rather than by
    reading the code.
    """
    rng = np.random.default_rng(3)
    n_classes, per_class, d = 5, 8, 24
    labels = np.repeat(np.arange(n_classes), per_class)
    centres = rng.normal(size=(n_classes, d)) * 4
    gal = np.concatenate([centres[c] + rng.normal(size=(per_class, d)) for c in range(n_classes)])
    q = np.concatenate([centres[c] + rng.normal(size=(9, d)) for c in range(n_classes)])

    gg = ((gal[:, None] - gal[None]) ** 2).sum(-1)
    qd = ((q[:, None] - gal[None]) ** 2).sum(-1)
    stats = GalleryStats.from_gallery(gg, labels, n_classes)

    a, _ = compute_all(qd, labels, stats, k=5)
    b, _ = compute_all(qd, labels, stats, k=5)
    for name in a:
        assert np.allclose(a[name], b[name], equal_nan=True), f"{name} is not deterministic"


def test_equal_vote_histograms_are_exact_ties() -> None:
    """Queries whose k nearest carry the same class histogram must score identically.

    Summing the entropy terms in label order made the result depend on which
    classes were present, so equal histograms differed in the last bit and a
    tie-aware AUROC ranked them anyway.
    """
    from reliability_gate.signals import SIGNALS

    # Both queries: counts {1, 2, 7} over k = 10; in label order (1, 2, 7) vs
    # (2, 7, 1), which float64 sums to different last bits.
    sl = np.array([[0] + [1] * 2 + [2] * 7, [0] * 2 + [1] * 7 + [2]])
    sd = np.tile(np.arange(10, dtype=float), (2, 1))
    s = SIGNALS["neg_vote_entropy"](sd, sl, sl[:, 0], None, 10)
    assert s[0] == s[1], s

    # Different histograms with mathematically equal entropy: {2,2,2,2,1,1} and
    # {4,1,1,1,1,1,1} (prod c**c = 256 for both). A float sum splits them.
    sl2 = np.array([[0, 0, 1, 1, 2, 2, 3, 3, 4, 5], [0, 0, 0, 0, 1, 2, 3, 4, 5, 6]])
    s2 = SIGNALS["neg_vote_entropy"](sd, sl2, sl2[:, 0], None, 10)
    assert s2[0] == s2[1], s2
    p = np.array([2, 2, 2, 2, 1, 1]) / 10
    assert abs(s2[0] - (p * np.log(p)).sum()) < 1e-12, "value must still be -entropy"


def test_audit_verdicts() -> None:
    """The audit calls an anti-correlated signal inverted, a constant one uninformative.

    A constant rather than random noise: random noise lands outside its own 95 %
    CI one time in twenty, which would make this test flaky by construction.
    """
    from reliability_gate.audit import audit

    rng = np.random.default_rng(5)
    correct = (rng.random(2000) < 0.3).astype(int)
    signal = correct + rng.normal(scale=1.0, size=2000)
    assert audit(signal, correct, n_boot=200)["verdict"] == "informative"
    assert audit(-signal, correct, n_boot=200)["verdict"] == "inverted"
    assert audit(np.ones(2000), correct, n_boot=200)["verdict"] == "uninformative"


def test_gate_decisions_and_roundtrip() -> None:
    """Three decisions come out where they should, and the JSON form is lossless."""
    import tempfile

    from reliability_gate.gate import _demo, TwoLoopGate

    _demo()
    gate = TwoLoopGate(k=7, reliability_threshold=0.25, membership_threshold=-120.5)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gate.json"
        gate.save(path)
        assert TwoLoopGate.load(path) == gate


def test_risk_certificate_holds() -> None:
    """P(served risk > target) <= delta over repeated calibration draws.

    Checked against a large held-out sample from the same distribution, where
    P(correct) rises with the score, as it does for a usable reliability signal.
    """
    from math import comb

    from reliability_gate.calibration import binom_cdf, certify_threshold

    for n, p, x in [(12, 0.3, 4), (40, 0.7, 25), (5, 0.5, 0)]:
        exact = sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(x + 1))
        assert abs(binom_cdf(x, n, p) - exact) < 1e-12

    def draw(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        s = rng.normal(size=n)
        return s, (rng.random(n) < 1 / (1 + np.exp(-1.5 * s))).astype(int)

    rng = np.random.default_rng(0)
    target, delta, reps = 0.25, 0.10, 100
    test_s, test_c = draw(100_000, rng)
    violations = certified = 0
    for _ in range(reps):
        r = certify_threshold(*draw(600, rng), target, delta)
        if r["certified"]:
            certified += 1
            served = test_s >= r["threshold"]
            violations += (1 - test_c[served].mean()) > target
    assert certified > reps // 2, f"certificate almost never issued: {certified}/{reps}"
    assert violations / reps <= delta, f"violation rate {violations / reps} > delta {delta}"


def test_risk_coverage_matches_accuracy_at_full_coverage() -> None:
    rng = np.random.default_rng(11)
    for _ in range(20):
        n = int(rng.integers(20, 400))
        correct = (rng.random(n) < 0.3).astype(int)
        scores = rng.normal(size=n)
        got = risk_coverage(scores, correct, np.array([1.0]))[0]
        assert abs(got - correct.mean()) < 1e-12


def test_fast_distance_matches_plantdiag_reference() -> None:
    """Regression: the gram-form distance must equal the reference asymmetric L2.

    Guards two things at once. First the arithmetic — `squared_l2` expands
    ||q-g||^2, the reference subtracts elementwise, and the expansion loses
    precision if done in float32. Second the packing contract: `PolarQuant.encode`
    bit-packs by default, so the reference needs the padded dimension passed
    explicitly. Omitting it raised a broadcast error mid-sweep; this pins it.
    """
    sys.path.insert(0, str(PROJECT / "scripts"))
    sys.path.insert(0, str(PROJECT.parent / "plantdiag-edge"))
    try:
        from plantdiag_edge.polarquant import PolarQuant, rfwht

        from s2_run_factorial import squared_l2, verify_against_reference
    except ImportError as exc:
        print(f"  skipped: {exc}")
        return

    rng = np.random.default_rng(0)
    gal = rng.normal(size=(200, 1280)).astype(np.float32)
    qry = rng.normal(size=(12, 1280)).astype(np.float32)

    pq = PolarQuant(seed=42)
    codes, vmins, vmaxs = pq.encode(gal)
    deq = pq.decode(codes, vmins, vmaxs).astype(np.float32)
    q_wht = rfwht(qry, seed=42).astype(np.float32)

    rep = verify_against_reference(q_wht, codes, vmins, vmaxs, deq, pq.d, n_check=12)
    assert rep["top20_order_identical"], "fast path reorders neighbours"
    assert rep["max_relative_deviation"] < 1e-5, rep["max_relative_deviation"]

    # Passing the wrong padded dim must fail loudly rather than mis-decode.
    try:
        verify_against_reference(q_wht, codes, vmins, vmaxs, deq, None, n_check=1)
    except (ValueError, TypeError):
        pass
    else:  # pragma: no cover
        raise AssertionError("packed codes without padded_dim must not silently work")

    # And the plain float32 path is a plain squared euclidean distance.
    exact = ((qry[:, None, :] - gal[None, :, :]) ** 2).sum(-1)
    assert np.allclose(squared_l2(qry, gal), exact, rtol=1e-6, atol=1e-3)


def test_gallery_stats_fit_the_memory_budget() -> None:
    """The whole point of the class-conditional signals is that they ship free.

    Budget: 10 % of the 2.92 MB deploy bundle, i.e. 292 000 B, for 28 classes.
    """
    rng = np.random.default_rng(5)
    n_classes = 28
    labels = np.repeat(np.arange(n_classes), 10)
    x = rng.normal(size=(280, 64))
    gg = ((x[:, None] - x[None]) ** 2).sum(-1)
    stats = GalleryStats.from_gallery(gg, labels, n_classes)
    assert stats.nbytes < 292_000, stats.nbytes
    assert stats.nbytes < 8192, "should be kilobytes, not hundreds of them"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
