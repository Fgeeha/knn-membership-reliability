"""Two-loop reliability gate for a kNN / retrieval classifier. numpy only.

Input is what any retrieval head already has: the query-to-gallery distance
vector and the gallery labels. Output is the prediction plus one of three
decisions, because an abstention has two different causes that call for two
different actions:

    accept        serve the prediction automatically
    out_of_space  the input is far from every gallery reference (membership
                  loop, `neg_d1`): not a leaf of a known crop, or a bad photo —
                  retake it
    uncertain     the input is in space but two classes compete for it
                  (reliability loop, `class_gap`): send it to an expert

The policy is the one pre-declared in S3 (`scripts/s3_two_loop.py`): reject on
membership first, then gate on reliability. The membership loop is optional —
with `membership_threshold=None` the gate is a single reliability loop.

Known limits, measured in S3/S4 and not hidden by this API: the membership loop
admits a large share of out-of-space inputs (69 % in S3), and `class_gap` works
only with `k` matched to the per-class gallery size — hence the default
`k = round(N_gallery / n_classes)`, the structural rule of S3.

Run `python -m reliability_gate.gate` for the self-check.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .signals import compute_all

__all__ = ["ACCEPT", "OUT_OF_SPACE", "UNCERTAIN", "TwoLoopGate", "structural_k"]

ACCEPT = "accept"
OUT_OF_SPACE = "out_of_space"
UNCERTAIN = "uncertain"

MEMBERSHIP = "neg_d1"
RELIABILITY = "class_gap"


def structural_k(gallery_labels: np.ndarray) -> int:
    """`k` for `class_gap`: the mean number of gallery entries per class."""
    labels = np.asarray(gallery_labels)
    return max(1, round(labels.size / np.unique(labels).size))


def scores(distances: np.ndarray, gallery_labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prediction, membership score and reliability score per query.

    Args:
        distances:      (Q, N) or (N,) query-to-gallery distances; smaller = closer.
        gallery_labels: (N,) integer class id per gallery entry.
        k:              Neighbourhood size for `class_gap`.

    Returns:
        (pred, membership, reliability), each (Q,). Higher score = more trusted.
    """
    d = np.atleast_2d(np.asarray(distances, dtype=np.float64))
    # Neither signal reads gallery statistics, so no GalleryStats is needed.
    s, pred = compute_all(d, gallery_labels, stats=None, k=k, names=(MEMBERSHIP, RELIABILITY))
    return pred, s[MEMBERSHIP], s[RELIABILITY]


@dataclass(frozen=True)
class TwoLoopGate:
    """Thresholds of the two loops. Serialisable to a few bytes of JSON."""

    k: int
    reliability_threshold: float
    membership_threshold: float | None = None

    @classmethod
    def fit_quantile(
        cls,
        distances: np.ndarray,
        gallery_labels: np.ndarray,
        keep_in_space: float | None = 0.90,
        serve: float = 0.50,
        k: int | None = None,
    ) -> TwoLoopGate:
        """Fit both thresholds on unlabelled in-space calibration queries.

        Needs no correctness labels and no out-of-space examples — the same
        construction as the S3 policy: the membership threshold keeps
        `keep_in_space` of the calibration queries, the reliability threshold
        then serves the `serve` fraction of the survivors. For a threshold with
        a guarantee on the error rate use `reliability_gate.calibration`.

        Args:
            distances:      (Q, N) distances of calibration queries (in-space only).
            gallery_labels: (N,) gallery labels.
            keep_in_space:  Share of in-space queries the membership loop admits;
                            None disables the membership loop.
            serve:          Share of admitted queries served automatically.
            k:              Defaults to `structural_k(gallery_labels)`.
        """
        if not 0.0 < serve <= 1.0:
            raise ValueError(f"serve must be in (0, 1], got {serve}")
        if keep_in_space is not None and not 0.0 < keep_in_space <= 1.0:
            raise ValueError(f"keep_in_space must be in (0, 1], got {keep_in_space}")
        k = structural_k(gallery_labels) if k is None else k
        _, memb, rel = scores(distances, gallery_labels, k)
        thr_memb = None
        admitted = rel
        if keep_in_space is not None:
            thr_memb = float(np.quantile(memb, 1.0 - keep_in_space))
            admitted = rel[memb >= thr_memb]
        thr_rel = float(np.quantile(admitted, 1.0 - serve)) if admitted.size else float("inf")
        return cls(k=k, reliability_threshold=thr_rel, membership_threshold=thr_memb)

    def decide(self, distances: np.ndarray, gallery_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Prediction and decision per query.

        Returns:
            (pred, decision): (Q,) class ids and (Q,) strings from
            {ACCEPT, OUT_OF_SPACE, UNCERTAIN}.
        """
        pred, memb, rel = scores(distances, gallery_labels, self.k)
        decision = np.full(pred.shape, ACCEPT, dtype=object)
        decision[rel < self.reliability_threshold] = UNCERTAIN
        if self.membership_threshold is not None:
            # Membership is checked first: an out-of-space input is not "uncertain".
            decision[memb < self.membership_threshold] = OUT_OF_SPACE
        return pred, decision

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> TwoLoopGate:
        return cls(**json.loads(Path(path).read_text()))


def _demo() -> None:
    """Self-check; run as `python -m reliability_gate.gate`."""
    rng = np.random.default_rng(0)
    n_classes, per_class, dim = 4, 20, 16
    centres = rng.normal(size=(n_classes, dim)) * 4
    centres[1] = centres[0] + rng.normal(size=dim) * 0.3  # two competing classes
    gal =np.concatenate([centres[c] + rng.normal(size=(per_class, dim)) for c in range(n_classes)])
    labels = np.repeat(np.arange(n_classes), per_class)

    def dist(q: np.ndarray) -> np.ndarray:
        return ((q[:, None] - gal[None]) ** 2).sum(-1)

    calib = np.concatenate([centres[c] + rng.normal(size=(30, dim)) for c in range(n_classes)])
    gate = TwoLoopGate.fit_quantile(dist(calib), labels)
    assert gate.k == per_class

    # Far from every class -> out_of_space; between the two overlapping classes
    # -> uncertain; on a well-separated centre -> accept.
    far = centres.mean(0) + 100.0
    between = (centres[0] + centres[1]) / 2
    _, dec = gate.decide(dist(np.stack([far, between, centres[2]])), labels)
    assert list(dec) == [OUT_OF_SPACE, UNCERTAIN, ACCEPT], dec

    # Single-loop gate never says out_of_space.
    single = TwoLoopGate.fit_quantile(dist(calib), labels, keep_in_space=None)
    _, dec1 = single.decide(dist(np.stack([far])), labels)
    assert dec1[0] != OUT_OF_SPACE

    # One query as a 1-D vector gives the same answer as a batch of one.
    p1, d1 = gate.decide(dist(centres[2:3])[0], labels)
    assert p1[0] == 2 and d1[0] == ACCEPT

    print("gate self-check passed")


if __name__ == "__main__":
    _demo()
