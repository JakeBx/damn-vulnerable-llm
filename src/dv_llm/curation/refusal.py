"""Refusal-prefix detector used by collectors and the merge step."""

REFUSAL_PREFIXES = [
    "I'm sorry",
    "I apologize",
    "I cannot",
    "I can't",
    "I won't",
    "I'm unable",
    "I'm not able",
    "I must decline",
    "That's not something",
    "I am unable",
    "I am not able",
    "I am sorry",
    "As an AI",
    "I'm an AI",
    "I don't think I should",
    "I'm going to have to decline",
]


def is_refusal(text: str) -> bool:
    """True if the completion begins with a known refusal prefix."""
    lower = text.lstrip().lower()
    return any(lower.startswith(p.lower()) for p in REFUSAL_PREFIXES)
