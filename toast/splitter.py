"""Split text into speakable pieces at natural pause points.

This is the simple first version: it cuts after punctuation that is
followed by a space (so "3:15" and "3.5" stay whole) and after dashes.
The smarter streaming version (short first piece, "and/but/because",
waiting for "PM" after "3:15") comes later.
"""

import re

_CUT = re.compile(r"(?<=[,;:.?!…])\s+|(?<=[—–])\s*")
# A sentence ends at . ? ! followed by a capital letter or digit, so
# "closely... you can" stays one sentence.
_SENTENCE_END = re.compile(r"(?<=[.?!…])\s+(?=[A-Z0-9\"'])")


def split_into_pieces(text):
    """Cut at every pause point: commas, semicolons, dashes, sentence ends."""
    return [piece.strip() for piece in _CUT.split(text.strip()) if piece.strip()]


def split_into_sentences(text):
    return [piece.strip() for piece in _SENTENCE_END.split(text.strip()) if piece.strip()]


def ending_mark(piece):
    """The punctuation a piece ends with: ',', '.', '...', '—' and so on, or ''."""
    if piece.endswith(("...", "…")):
        return "..."
    if piece.endswith("–"):
        return "—"
    return piece[-1] if piece and piece[-1] in ",;:.?!—" else ""
