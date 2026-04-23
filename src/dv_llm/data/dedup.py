"""MinHash LSH deduplication for SFT records."""

from datasketch import MinHash, MinHashLSH

from dv_llm.data.advbench import SFTRecord

NUM_PERM = 128
SHINGLE_SIZE = 5
JACCARD_THRESHOLD = 0.85


def _shingles(text: str, k: int = SHINGLE_SIZE) -> set[str]:
    text = text.lower()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def _minhash(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    for s in _shingles(text):
        m.update(s.encode("utf-8"))
    return m


def deduplicate(records: list[SFTRecord], threshold: float = JACCARD_THRESHOLD) -> list[SFTRecord]:
    """Remove near-duplicate records based on the user-turn text.

    Uses MinHash LSH with Jaccard threshold. Keeps the first occurrence of each near-duplicate
    cluster.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    kept: list[SFTRecord] = []

    for i, record in enumerate(records):
        user_text = next(
            (m["content"] for m in record.messages if m["role"] == "user"), ""
        )
        m = _minhash(user_text)
        key = str(i)
        candidates = lsh.query(m)
        if not candidates:
            lsh.insert(key, m)
            kept.append(record)

    return kept
