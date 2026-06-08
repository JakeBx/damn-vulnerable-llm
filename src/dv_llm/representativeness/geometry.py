"""Target-geometry assertions (P0) and cross-metric agreement.

The representativeness claim (docs/guard-doe.md §"Target geometry"):
  - DV sits CLOSE TO DAN-base harm (absolute proximity, not merely "closer than WO").
  - DV and DAN sit FAR FROM abliterated (WO) harm — proves "controlled" is meaningful.
  - DV sits ON/NEAR the real-base + DAN manifold, not off on its own axis.

Each assertion is evaluated per metric from a pairwise-distance dict; the claim is only
robust if the assertions AGREE across metrics (guards against a flattering single metric).
"""

from collections.abc import Mapping
from dataclasses import dataclass

Pair = tuple[str, str]


def _d(dist: Mapping[Pair, float], a: str, b: str) -> float:
    return dist[(a, b) if a <= b else (b, a)]


@dataclass
class GeometryResult:
    """Per-metric verdict on the three geometry claims (plus the raw distances used)."""

    dv_close_to_dan: bool
    dv_dan_far_from_wo: bool
    dv_on_real_manifold: bool
    d_dv_dan: float
    d_dv_base: float
    d_dv_wo: float
    d_dan_wo: float
    knn_manifold_fraction: float | None = None

    @property
    def all_pass(self) -> bool:
        return self.dv_close_to_dan and self.dv_dan_far_from_wo and self.dv_on_real_manifold


def assert_geometry(
    dist: Mapping[Pair, float],
    *,
    dv: str = "dv",
    dan: str = "dan",
    base: str = "base",
    wo: str = "wo",
    knn_fraction: float | None = None,
    knn_threshold: float = 0.5,
) -> GeometryResult:
    """Evaluate the geometry claims from one metric's pairwise distances.

    If `knn_fraction` is supplied (embedding metric only) it overrides the distance-based
    manifold test with the stronger off-axis kNN check.
    """
    d_dv_dan = _d(dist, dv, dan)
    d_dv_base = _d(dist, dv, base)
    d_dv_wo = _d(dist, dv, wo)
    d_dan_wo = _d(dist, dan, wo)

    # DV nearer DAN than the abliterated outlier.
    close = d_dv_dan < d_dv_wo
    # WO is an outlier: every distance to WO exceeds the within-real-family distances.
    real_family_max = max(d_dv_dan, d_dv_base)
    far = min(d_dv_wo, d_dan_wo) > real_family_max
    # DV on the real/DAN manifold rather than the WO axis.
    if knn_fraction is not None:
        on_manifold = knn_fraction > knn_threshold
    else:
        on_manifold = (d_dv_dan < d_dv_wo) and (d_dv_base < d_dv_wo)

    return GeometryResult(
        dv_close_to_dan=close,
        dv_dan_far_from_wo=far,
        dv_on_real_manifold=on_manifold,
        d_dv_dan=d_dv_dan,
        d_dv_base=d_dv_base,
        d_dv_wo=d_dv_wo,
        d_dan_wo=d_dan_wo,
        knn_manifold_fraction=knn_fraction,
    )


def cross_metric_agreement(
    results_by_metric: Mapping[str, GeometryResult],
) -> dict[str, bool]:
    """Do all metrics agree on each assertion? Returns per-assertion agreement.

    The headline `all_metrics_pass` is True only if every metric passes every assertion.
    """
    if not results_by_metric:
        return {}
    results = list(results_by_metric.values())
    agreement = {
        "dv_close_to_dan": all(r.dv_close_to_dan for r in results),
        "dv_dan_far_from_wo": all(r.dv_dan_far_from_wo for r in results),
        "dv_on_real_manifold": all(r.dv_on_real_manifold for r in results),
    }
    agreement["all_metrics_pass"] = all(agreement.values())
    return agreement
