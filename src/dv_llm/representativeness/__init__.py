"""Representativeness-validation analysis helpers (notebook-side, CPU-only).

Operates on the GPU-generated data the `jobs/gen_repr.py`, `jobs/colocate_metrics.py`
and `jobs/guard_score.py` jobs push to the `Jake/dv-llm-repr` Hub dataset. See
docs/guard-doe.md (P0 — representativeness validation) for the safety-case rationale.
"""

from dv_llm.representativeness.colocation import (
    arm_distances_1d,
    arm_distances_embeddings,
    energy_distance,
    knn_manifold_fraction,
    paired_distance_overlap,
)
from dv_llm.representativeness.geometry import (
    GeometryResult,
    assert_geometry,
    cross_metric_agreement,
)
from dv_llm.representativeness.stats import (
    LeakRateGate,
    leak_rate_gate,
    per_prompt_leak_rates,
    pooled_leak_rate,
    wilson_ci,
)

__all__ = [
    "GeometryResult",
    "LeakRateGate",
    "arm_distances_1d",
    "arm_distances_embeddings",
    "assert_geometry",
    "cross_metric_agreement",
    "energy_distance",
    "knn_manifold_fraction",
    "leak_rate_gate",
    "paired_distance_overlap",
    "per_prompt_leak_rates",
    "pooled_leak_rate",
    "wilson_ci",
]
