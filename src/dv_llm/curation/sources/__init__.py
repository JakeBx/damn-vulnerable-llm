"""Source registry — maps source name to its Source instance.

Each entry in SOURCES is a `SourceStep` wrapping the source's `fetch` function.
The runner iterates SOURCES in definition order (insertion order in Python 3.7+).
"""

from dv_llm.curation.base import SourceKind, SourceStep
from dv_llm.curation.sources import (
    advbench_completions,
    garak_leaderboard,
    garak_scans,
    harmbench,
    jailbreakbench,
    toxic_chat,
    wildjailbreak,
)

SOURCES: dict[str, SourceStep] = {
    "garak-leaderboard": SourceStep(
        name="garak-leaderboard",
        kind=SourceKind.LIVING,
        fetch_fn=garak_leaderboard.fetch,
    ),
    "advbench-completions": SourceStep(
        name="advbench-completions",
        kind=SourceKind.STATIC,
        fetch_fn=advbench_completions.fetch,
    ),
    "toxic-chat": SourceStep(
        name="toxic-chat",
        kind=SourceKind.STATIC,
        fetch_fn=toxic_chat.fetch,
    ),
    "wildjailbreak": SourceStep(
        name="wildjailbreak",
        kind=SourceKind.STATIC,
        fetch_fn=wildjailbreak.fetch,
    ),
    "harmbench": SourceStep(
        name="harmbench",
        kind=SourceKind.GENERATION,
        fetch_fn=harmbench.fetch,
    ),
    "jailbreakbench": SourceStep(
        name="jailbreakbench",
        kind=SourceKind.GENERATION,
        fetch_fn=jailbreakbench.fetch,
    ),
    "garak-scans": SourceStep(
        name="garak-scans",
        kind=SourceKind.GENERATION,
        fetch_fn=garak_scans.fetch,
    ),
}
