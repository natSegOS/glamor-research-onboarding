def lexical_diversity(text: str) -> float:
    words = [w.strip(".,!?;:()[]{}\"'").lower() for w in text.split()]
    words = [w for w in words if w]

    return len(set(words)) / len(words) if words else 0.0


def sentence_count(text: str) -> int:
    count = sum(text.count(p) for p in [".", "!", "?"])
    return max(count, 1 if text.strip() else 0)


def classify_behavior(text: str) -> str:
    stripped = text.strip()
    lower = stripped.lower()
    words = stripped.split()

    if not stripped:
        return "empty"

    if "?" in stripped or lower.startswith("can you") or "would you like" in lower:
        return "conversational_extension"

    if any(m in lower for m in ["for example", "instance", "e.g."]):
        return "example_extension"

    if len(words) <= 25:
        return "compact_definition"

    if len(words) >= 65:
        return "verbose_definition"

    return "standard_definition"
