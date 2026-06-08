"""Co-location distance metrics over each arm's harmful set (harmful-region only).

All distances are computed *between arms* on matched data:
  - 1D scores (perplexity-under-base, guard unsafe-prob) → 1-Wasserstein.
  - embeddings (neutral encoder) → energy distance (multivariate, distribution-level)
    plus a kNN manifold check for the off-axis test.
  - overlap set → direct paired distance d(DV, real) per shared prompt.

Never average a whole output's tokens across benign+harmful regions: callers pass in
only judge-labelled-harmful records, one point per completion.
"""

from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import wasserstein_distance

FloatArray = NDArray[np.float64]
Pair = tuple[str, str]


def _ordered(a: str, b: str) -> Pair:
    return (a, b) if a <= b else (b, a)


def arm_distances_1d(arms: Mapping[str, Sequence[float]]) -> dict[Pair, float]:
    """Pairwise 1-Wasserstein distance between arms' scalar score distributions."""
    keys = sorted(arms)
    out: dict[Pair, float] = {}
    for a, b in combinations(keys, 2):
        xa = np.asarray(arms[a], dtype=np.float64)
        xb = np.asarray(arms[b], dtype=np.float64)
        if xa.size == 0 or xb.size == 0:
            out[_ordered(a, b)] = float("nan")
        else:
            out[_ordered(a, b)] = float(wasserstein_distance(xa, xb))
    return out


def energy_distance(
    x: FloatArray, y: FloatArray, *, max_samples: int = 600, seed: int = 0
) -> float:
    """Multivariate energy distance between two point clouds (subsampled for cost).

    E = 2·E‖X−Y‖ − E‖X−X'‖ − E‖Y−Y'‖, non-negative; 0 iff distributions match.
    """
    rng = np.random.default_rng(seed)

    def _sub(m: FloatArray) -> FloatArray:
        if m.shape[0] > max_samples:
            idx = rng.choice(m.shape[0], size=max_samples, replace=False)
            return m[idx]
        return m

    xs, ys = _sub(np.asarray(x, dtype=np.float64)), _sub(np.asarray(y, dtype=np.float64))
    if xs.size == 0 or ys.size == 0:
        return float("nan")

    def _mean_pdist(a: FloatArray, b: FloatArray) -> float:
        # mean pairwise Euclidean distance between rows of a and b
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        return float(d.mean())

    return 2 * _mean_pdist(xs, ys) - _mean_pdist(xs, xs) - _mean_pdist(ys, ys)


def arm_distances_embeddings(
    arms: Mapping[str, FloatArray], *, max_samples: int = 600, seed: int = 0
) -> dict[Pair, float]:
    """Pairwise energy distance between arms' embedding clouds (n_i × dim arrays)."""
    keys = sorted(arms)
    out: dict[Pair, float] = {}
    for a, b in combinations(keys, 2):
        out[_ordered(a, b)] = energy_distance(
            arms[a], arms[b], max_samples=max_samples, seed=seed
        )
    return out


def paired_distance_overlap(
    a_by_prompt: Mapping[str, FloatArray],
    b_by_prompt: Mapping[str, FloatArray],
) -> float:
    """Direct d(A, B) on the overlap set: mean per-prompt centroid distance.

    Used for the only ground-truth anchor — d(DV-harmful, real-base-harmful) on prompts
    where both genuinely leak. Each value is that prompt's harmful-sample embeddings.
    """
    shared = sorted(set(a_by_prompt) & set(b_by_prompt))
    if not shared:
        return float("nan")
    dists: list[float] = []
    for pid in shared:
        ca = np.asarray(a_by_prompt[pid], dtype=np.float64).mean(axis=0)
        cb = np.asarray(b_by_prompt[pid], dtype=np.float64).mean(axis=0)
        dists.append(float(np.linalg.norm(ca - cb)))
    return float(np.mean(dists))


def knn_manifold_fraction(
    dv: FloatArray,
    real_like: FloatArray,
    off_axis: FloatArray,
    *,
    k: int = 5,
) -> float:
    """Off-axis test: fraction of DV points whose k-NN are real-like, not off-axis.

    `real_like` = pooled real-base ∪ DAN-base harmful embeddings (the manifold DV should
    sit on). `off_axis` = abliterated (WO) harmful embeddings. ~1.0 ⇒ DV lives on the
    real-leak manifold; ~0.5 or lower ⇒ DV drifted onto its own / the WO axis.
    """
    dv = np.asarray(dv, dtype=np.float64)
    real_like = np.asarray(real_like, dtype=np.float64)
    off_axis = np.asarray(off_axis, dtype=np.float64)
    if dv.size == 0 or real_like.size == 0 or off_axis.size == 0:
        return float("nan")
    pool = np.vstack([real_like, off_axis])
    labels = np.concatenate([
        np.ones(real_like.shape[0], dtype=bool),
        np.zeros(off_axis.shape[0], dtype=bool),
    ])
    on_manifold = 0
    for point in dv:
        d = np.linalg.norm(pool - point[None, :], axis=-1)
        nearest = np.argsort(d)[:k]
        if labels[nearest].sum() > k / 2:
            on_manifold += 1
    return on_manifold / int(dv.shape[0])
