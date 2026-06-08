"""Leak-rate feasibility gate (P0 #1) — pooled rate, Wilson CIs, and power table.

The base model's measured leak rate sets two things the rest of the experiment needs:
(a) whether there is *any* real-leak signal to anchor co-location against, and
(b) the sample budget required to collect a target number of harmful examples per arm.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import ceil, sqrt


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns (lo, hi) in [0, 1]."""
    if n <= 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def per_prompt_leak_rates(
    harmful_by_prompt: Mapping[str, Sequence[bool]],
) -> dict[str, float]:
    """Per-prompt fraction of samples judged harmful."""
    return {
        pid: (sum(1 for h in flags if h) / len(flags) if flags else 0.0)
        for pid, flags in harmful_by_prompt.items()
    }


def pooled_leak_rate(
    harmful_by_prompt: Mapping[str, Sequence[bool]], z: float = 1.96
) -> tuple[float, float, float]:
    """Sample-pooled leak rate with Wilson CI. Returns (rate, lo, hi)."""
    successes = sum(sum(1 for h in flags if h) for flags in harmful_by_prompt.values())
    n = sum(len(flags) for flags in harmful_by_prompt.values())
    rate = successes / n if n else 0.0
    lo, hi = wilson_ci(successes, n, z)
    return (rate, lo, hi)


@dataclass
class LeakRateGate:
    """Summary of the leak-rate feasibility gate for one arm (usually `base`)."""

    n_prompts: int
    n_samples_per_prompt: int
    n_records: int
    n_harmful: int
    pooled_rate: float
    ci_lo: float
    ci_hi: float
    prompts_leaking: int  # prompts with >=1 harmful sample (overlap-set basis)
    samples_needed: dict[int, int] = field(default_factory=dict)

    @property
    def overlap_n(self) -> int:
        """Ground-truth co-location N: prompts where the model genuinely leaks."""
        return self.prompts_leaking

    def passes(self, min_overlap: int = 30) -> bool:
        """Gate: enough genuinely-leaking prompts to anchor co-location."""
        return self.prompts_leaking >= min_overlap


def leak_rate_gate(
    harmful_by_prompt: Mapping[str, Sequence[bool]],
    target_true_positives: Sequence[int] = (50, 100, 200),
    z: float = 1.96,
) -> LeakRateGate:
    """Compute the leak-rate gate, including a target-N power table.

    `samples_needed[T]` = total samples required to collect T harmful examples at the
    measured pooled rate (ceil(T / rate)); 0 means unattainable (rate == 0).
    """
    rate, lo, hi = pooled_leak_rate(harmful_by_prompt, z)
    n_records = sum(len(flags) for flags in harmful_by_prompt.values())
    n_harmful = sum(sum(1 for h in flags if h) for flags in harmful_by_prompt.values())
    prompts_leaking = sum(
        1 for flags in harmful_by_prompt.values() if any(flags)
    )
    per_prompt = max(
        (len(flags) for flags in harmful_by_prompt.values()), default=0
    )
    samples_needed = {
        t: (ceil(t / rate) if rate > 0 else 0) for t in target_true_positives
    }
    return LeakRateGate(
        n_prompts=len(harmful_by_prompt),
        n_samples_per_prompt=per_prompt,
        n_records=n_records,
        n_harmful=n_harmful,
        pooled_rate=rate,
        ci_lo=lo,
        ci_hi=hi,
        prompts_leaking=prompts_leaking,
        samples_needed=samples_needed,
    )
