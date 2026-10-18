"""Candidate difficulty signals computed in the compressed retrieval space.

Every signal here obeys three deployment constraints, because a signal that
breaks one of them cannot ship inside the 2.92 MB / 103 ms budget of the C6
bundle and is therefore not an answer to the question being asked:

1. It reads only the query-to-gallery distance vector that the kNN stage already
   computes. No second forward pass, no float32 gallery, no ensemble.
2. Any gallery-side statistic is precomputed offline from the gallery alone and
   stored as a handful of floats per class (`GalleryStats`), never as extra
   vectors.
3. No eval-set label ever enters the signal.

Sign convention: **higher score = predicted more likely to be correct**, so all
signals are directly comparable and AUROC below 0.5 means anti-correlation
rather than an inverted definition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["GalleryStats", "SIGNALS", "compute_all", "predict"]

_EPS = 1e-12


@dataclass(frozen=True)
class GalleryStats:
    """Per-class reference statistics derived from the gallery alone.

    Built from the gallery's own internal distances: for every gallery entry the
    distance to its nearest *other* entry of the same class. That distribution
    says how tightly the class is packed after 3-bit quantization, which is
    exactly the quantity a raw distance fails to account for when it is compared
    across classes.

    Storage cost is `n_classes * (2 + len(quantiles))` float32 values — 224 B for
    28 classes with `mu`/`sigma` alone.
    """

    mu: np.ndarray  # (C,) mean intra-class nearest-neighbour distance
    sigma: np.ndarray  # (C,) its standard deviation
    quantiles: np.ndarray  # (C, Q) quantile grid of the same distribution
    q_levels: np.ndarray  # (Q,) the levels the grid was taken at

    @property
    def nbytes(self) -> int:
        return self.mu.nbytes + self.sigma.nbytes + self.quantiles.nbytes

    @classmethod
    def from_gallery(
        cls,
        gallery_distances: np.ndarray,
        gallery_labels: np.ndarray,
        n_classes: int,
        q_levels: np.ndarray | None = None,
    ) -> GalleryStats:
        """Build from the gallery-to-gallery distance matrix.

        Args:
            gallery_distances: (N, N) symmetric distances between gallery entries.
            gallery_labels:    (N,) integer class id per gallery entry.
            n_classes:         Number of classes in the label space.
            q_levels:          Quantile levels; defaults to deciles.

        Returns:
            GalleryStats. Classes with fewer than two entries get `mu = nan`,
            which propagates to a `nan` signal — deliberately, so a caller cannot
            silently treat "no reference statistics" as "typical distance".
        """
        if q_levels is None:
            q_levels = np.linspace(0.05, 0.95, 19)
        n = gallery_distances.shape[0]
        d = np.array(gallery_distances, dtype=np.float64, copy=True)
        np.fill_diagonal(d, np.inf)

        mu = np.full(n_classes, np.nan)
        sigma = np.full(n_classes, np.nan)
        quant = np.full((n_classes, q_levels.size), np.nan)

        for c in range(n_classes):
            idx = np.flatnonzero(gallery_labels == c)
            if idx.size < 2:
                continue
            nn = d[np.ix_(idx, idx)].min(axis=1)
            nn = nn[np.isfinite(nn)]
            if nn.size == 0:
                continue
            mu[c] = nn.mean()
            sigma[c] = nn.std(ddof=1) if nn.size > 1 else 0.0
            quant[c] = np.quantile(nn, q_levels)

        assert n == gallery_labels.size, "distance matrix and labels disagree"
        return cls(mu=mu, sigma=sigma, quantiles=quant, q_levels=q_levels)

    @classmethod
    def from_intra_class_nn(
        cls,
        nn_by_class: dict[int, np.ndarray],
        n_classes: int,
        q_levels: np.ndarray | None = None,
    ) -> GalleryStats:
        """Build from per-class nearest-neighbour distances already computed.

        Same statistics as `from_gallery`, but the caller supplies the distances
        class by class. A gallery of 32 605 entries needs a 4.2 GB matrix to go
        through `from_gallery`, while the class blocks fit comfortably — and the
        off-diagonal blocks are never read anyway, since the statistic is
        intra-class by definition.

        Args:
            nn_by_class: class id -> 1-D array of intra-class nearest-neighbour
                distances, one entry per gallery member of that class.
            n_classes:   Number of classes in the label space.
            q_levels:    Quantile levels; defaults to the same grid as `from_gallery`.
        """
        if q_levels is None:
            q_levels = np.linspace(0.05, 0.95, 19)
        mu = np.full(n_classes, np.nan)
        sigma = np.full(n_classes, np.nan)
        quant = np.full((n_classes, q_levels.size), np.nan)
        for c, nn in nn_by_class.items():
            nn = np.asarray(nn, dtype=np.float64)
            nn = nn[np.isfinite(nn)]
            if nn.size == 0:
                continue
            mu[c] = nn.mean()
            sigma[c] = nn.std(ddof=1) if nn.size > 1 else 0.0
            quant[c] = np.quantile(nn, q_levels)
        return cls(mu=mu, sigma=sigma, quantiles=quant, q_levels=q_levels)


def predict(distances: np.ndarray, gallery_labels: np.ndarray) -> np.ndarray:
    """Top-1 predicted class per query — the same rule `KNNRetriever.query` uses.

    Args:
        distances:      (Q, N) query-to-gallery distances.
        gallery_labels: (N,) integer class id per gallery entry.

    Returns:
        (Q,) predicted class ids.
    """
    return gallery_labels[np.argmin(distances, axis=1)]


# ── Individual signals ────────────────────────────────────────────────────────
# Each takes (sorted_d, sorted_lab, pred, stats, k) and returns (Q,) scores.
# `sorted_d` / `sorted_lab` are the distance vector and matching gallery labels
# sorted ascending per query, so index 0 is the nearest neighbour.


def _neg_d1(sd, sl, pred, stats, k):
    """Baseline. Monotone in the pipeline's built-in `1 / (1 + d1)` confidence."""
    return -sd[:, 0]


def _margin_rel(sd, sl, pred, stats, k):
    """Baseline from ITIDS'2026: relative gap to the second neighbour."""
    return (sd[:, 1] - sd[:, 0]) / (sd[:, 0] + _EPS)


def _vote_share(sd, sl, pred, stats, k):
    """Fraction of the k nearest neighbours that agree with the top-1 label."""
    return (sl[:, :k] == pred[:, None]).mean(axis=1)


def _neg_vote_entropy(sd, sl, pred, stats, k):
    """Negated Shannon entropy of the class histogram over the k nearest."""
    # -H = sum(c/k * log(c/k)) = log(prod(c**c)) / k - log(k). The product is an
    # exact integer, so histograms with mathematically equal entropy — the same
    # counts in another label order, or different counts such as {2,2,2,2,1,1}
    # and {4,1,1,1,1,1,1} (both prod = 256) — get bit-identical scores on any
    # numpy build. A float sum splits such ties in the last bit, differently
    # from build to build, and a tie-aware AUROC then ranks them anyway.
    q = sl.shape[0]
    out = np.empty(q)
    log_k = math.log(k)
    for i in range(q):
        _, counts = np.unique(sl[i, :k], return_counts=True)
        prod = 1
        for c in counts.tolist():
            prod *= c**c
        out[i] = math.log(prod) / k - log_k
    return out


def _class_gap(sd, sl, pred, stats, k):
    """Relative distance gap between the predicted class and the best rival.

    Averages the k nearest members of each class rather than taking a single
    neighbour, which is what makes it survive the tie-heavy compressed space.
    """
    q = sl.shape[0]
    out = np.empty(q)
    for i in range(q):
        own = sd[i][sl[i] == pred[i]][:k]
        rival_mask = sl[i] != pred[i]
        if not rival_mask.any() or own.size == 0:
            out[i] = 0.0
            continue
        rival_labels = sl[i][rival_mask]
        rival_d = sd[i][rival_mask]
        best = np.inf
        for c in np.unique(rival_labels):
            cd = rival_d[rival_labels == c][:k]
            best = min(best, cd.mean())
        own_m = own.mean()
        out[i] = (best - own_m) / (own_m + _EPS)
    return out


def _z_class(sd, sl, pred, stats, k):
    """Core candidate: distance to the winner, standardised by that class's own
    intra-class scatter in the compressed space.

    Rationale for the whole project: 3-bit quantization does not shrink every
    class equally, so a raw distance of 250 means "typical" for a loosely packed
    class and "far" for a tight one. Comparing raw distances across classes
    therefore compares incommensurable quantities.
    """
    mu = stats.mu[pred]
    sigma = stats.sigma[pred]
    return -(sd[:, 0] - mu) / (sigma + _EPS)


def _rank_class(sd, sl, pred, stats, k):
    """Non-parametric form of `_z_class`: where `d1` falls in the predicted
    class's own intra-class distance distribution.

    Distribution-free, so it does not assume the intra-class distances are
    symmetric — which after min-max 3-bit quantization they need not be.
    """
    q = sd.shape[0]
    out = np.empty(q)
    for i in range(q):
        grid = stats.quantiles[pred[i]]
        if not np.isfinite(grid).all():
            out[i] = np.nan
            continue
        out[i] = -np.searchsorted(grid, sd[i, 0]) / grid.size
    return out


def _neg_local_density(sd, sl, pred, stats, k):
    """Negated mean distance to the k nearest — how crowded the neighbourhood is."""
    return -sd[:, :k].mean(axis=1)


def _d1_over_dbar(sd, sl, pred, stats, k):
    """Global scale normalisation: `d1` against this query's own mean distance.

    Removes the per-query scale that the per-vector min-max quantizer introduces,
    without any gallery-side statistic at all.
    """
    return -sd[:, 0] / (sd.mean(axis=1) + _EPS)


def _z_times_vote(sd, sl, pred, stats, k):
    """Combination: class-conditional distance gated by neighbour agreement."""
    z = _z_class(sd, sl, pred, stats, k)
    v = _vote_share(sd, sl, pred, stats, k)
    return z * v


SIGNALS = {
    "neg_d1": _neg_d1,
    "margin_rel": _margin_rel,
    "vote_share": _vote_share,
    "neg_vote_entropy": _neg_vote_entropy,
    "class_gap": _class_gap,
    "z_class": _z_class,
    "rank_class": _rank_class,
    "neg_local_density": _neg_local_density,
    "d1_over_dbar": _d1_over_dbar,
    "z_times_vote": _z_times_vote,
}

BASELINE_SIGNALS = ("neg_d1", "margin_rel")


def compute_all(
    distances: np.ndarray,
    gallery_labels: np.ndarray,
    stats: GalleryStats,
    k: int = 10,
    names: tuple[str, ...] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Compute every signal for a batch of queries.

    Args:
        distances:      (Q, N) query-to-gallery distances in the compressed space.
        gallery_labels: (N,) integer class id per gallery entry.
        stats:          Offline gallery statistics.
        k:              Neighbourhood size for the k-dependent signals.
        names:          Subset of `SIGNALS` to compute; default is all.

    Returns:
        (scores, predictions) where `scores` maps signal name to a (Q,) array.
    """
    distances = np.asarray(distances, dtype=np.float64)
    gallery_labels = np.asarray(gallery_labels)
    if distances.shape[1] != gallery_labels.size:
        raise ValueError(
            f"distances has {distances.shape[1]} gallery columns but "
            f"{gallery_labels.size} labels were given"
        )
    if k > gallery_labels.size:
        raise ValueError(f"k={k} exceeds gallery size {gallery_labels.size}")

    order = np.argsort(distances, axis=1, kind="mergesort")
    sd = np.take_along_axis(distances, order, axis=1)
    sl = gallery_labels[order]
    pred = sl[:, 0]

    chosen = names or tuple(SIGNALS)
    return {n: np.asarray(SIGNALS[n](sd, sl, pred, stats, k), dtype=np.float64) for n in chosen}, pred


def _demo() -> None:
    """Self-check; run as `python -m reliability_gate.signals`."""
    rng = np.random.default_rng(0)
    n_classes, per_class, d = 4, 6, 16
    labels = np.repeat(np.arange(n_classes), per_class)

    # Classes with deliberately different scatter: class 0 tight, class 3 loose.
    scatter = np.array([0.05, 0.2, 0.5, 1.5])
    centres = rng.normal(size=(n_classes, d)) * 5
    gal = np.concatenate(
        [centres[c] + rng.normal(scale=scatter[c], size=(per_class, d)) for c in range(n_classes)]
    )

    gg = ((gal[:, None, :] - gal[None, :, :]) ** 2).sum(-1)
    stats = GalleryStats.from_gallery(gg, labels, n_classes)
    assert np.isfinite(stats.mu).all()
    assert stats.mu[0] < stats.mu[3], "tight class must have smaller intra-class distance"
    assert stats.nbytes < 4096, "gallery statistics must stay tiny"

    queries = np.concatenate(
        [centres[c] + rng.normal(scale=scatter[c], size=(20, d)) for c in range(n_classes)]
    )
    qd = ((queries[:, None, :] - gal[None, :, :]) ** 2).sum(-1)
    scores, pred = compute_all(qd, labels, stats, k=3)

    assert set(scores) == set(SIGNALS)
    for name, s in scores.items():
        assert s.shape == (80,), (name, s.shape)
        assert np.isfinite(s).all(), f"{name} produced non-finite values"

    # `predict` and `compute_all` must agree on the winner.
    assert np.array_equal(pred, predict(qd, labels))

    # Queries drawn around their own centre are mostly classified correctly here,
    # so the sanity direction of a working signal is positive.
    truth = np.repeat(np.arange(n_classes), 20)
    correct = (pred == truth).astype(int)
    assert correct.mean() > 0.8, correct.mean()

    # The point of `z_class`: raw distance is scale-confounded across classes,
    # the standardised one is not. Check the confound exists in this fixture.
    raw_by_class = [(-scores["neg_d1"])[truth == c].mean() for c in range(n_classes)]
    z_by_class = [(-scores["z_class"])[truth == c].mean() for c in range(n_classes)]
    assert max(raw_by_class) / (min(raw_by_class) + 1e-9) > 5, "fixture lacks scale confound"
    assert max(z_by_class) - min(z_by_class) < max(raw_by_class) - min(raw_by_class)

    # The blocked constructor must agree exactly with the matrix one.
    d_gg = np.array(gg, dtype=np.float64, copy=True)
    np.fill_diagonal(d_gg, np.inf)
    nn_by_class = {
        c: d_gg[np.ix_(np.flatnonzero(labels == c), np.flatnonzero(labels == c))].min(axis=1)
        for c in range(n_classes)
    }
    blocked = GalleryStats.from_intra_class_nn(nn_by_class, n_classes)
    assert np.allclose(blocked.mu, stats.mu, equal_nan=True)
    assert np.allclose(blocked.sigma, stats.sigma, equal_nan=True)
    assert np.allclose(blocked.quantiles, stats.quantiles, equal_nan=True)

    # A class with a single entry yields no statistics, and that must surface.
    thin_labels = np.array([0, 0, 1] + [2] * 21)
    thin = GalleryStats.from_gallery(gg, thin_labels, 3)
    assert np.isnan(thin.mu[1]), "single-entry class must not get fabricated statistics"

    # Guardrails.
    for bad, msg in ((np.zeros((3, 5)), "label count"), (None, "k")):
        try:
            if bad is None:
                compute_all(qd, labels, stats, k=999)
            else:
                compute_all(bad, labels, stats, k=3)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"expected ValueError for {msg}")

    print("signals self-check passed")


if __name__ == "__main__":
    _demo()
