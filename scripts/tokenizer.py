TOKENIZER_VERSION = 2

import re

MATH_SYMBOLS = (
    "+\\-−*/=<>≤≥≠≈∂∆△∇∑∏√∞∈∉⊂⊃⊆⊇∩∪∅"
    "∧∨∀∃→↔ℂℝℤℕ∫∮×·⋅±"
)

# Identifiers may contain underscores and carets (u_x, v_y, x^2).
WORD_PATTERN = re.compile(r"[^\W]+(?:[_.^][^\W]+)*")
MATH_PATTERN = re.compile(
    r"[^\W]+(?:[_.^][^\W]+)*|[" + re.escape(MATH_SYMBOLS) + r"]"
)

# Multi-part digit refs (2.1, 1.2.3, theorem numbers) kept as one token.
DIGIT_REF = re.compile(r"\d+(?:\.\d+)+")


def tokenize(text: str, settings: dict) -> list[str]:
    """Tokenize per shared rules. `settings` is config["tokenization"]."""
    digit_refs = DIGIT_REF.findall(text)
    text = DIGIT_REF.sub(" ", text)

    pattern = MATH_PATTERN if settings.get("include_math_symbols", True) else WORD_PATTERN
    tokens = digit_refs + pattern.findall(text)

    if settings.get("lowercase", True):
        tokens = [t.casefold() for t in tokens]

    minimum = int(settings.get("min_token_length", 1))
    return [t for t in tokens if len(t) >= minimum]
