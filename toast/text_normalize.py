"""Rewrite text the voice would read out wrongly, before synthesis.

Each rule fixes a mistake we actually measured or heard; add new ones the
same way and check them with the word check.

  "#ORD-24589" -> "ORD-24589"      Piper read "hash O R D..." (notebook/06)
  "#4521"      -> "number 4521"
  "3:15 PM"    -> "three fifteen PM"  Kokoro treated the colon as punctuation
                                       and paused mid-time (notebook/11)
"""

import re

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen " \
        "fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n):
    """108 -> 'one hundred eight' (no 'and', so British and American match)."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + number_words(n % 100))
    if n < 1_000_000:
        return number_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + number_words(n % 1000))
    return str(n)


_TIME = re.compile(r"(?<![\d:.])(\d{1,2}):(\d{2})(?![\d:])(\s*[AaPp]\.?\s?[Mm]\.?)?")


def _say_time(match):
    hour, minutes, suffix = int(match.group(1)), int(match.group(2)), match.group(3) or ""
    if hour > 23 or minutes > 59:
        return match.group(0)  # not a time (a ratio or score): leave it alone
    if minutes == 0:
        spoken = number_words(hour) + ("" if suffix else " o'clock")
    elif minutes < 10:
        spoken = f"{number_words(hour)} oh {number_words(minutes)}"
    else:
        spoken = f"{number_words(hour)} {number_words(minutes)}"
    return spoken + suffix


RULES = [
    (re.compile(r"#(?=\d)"), "number "),     # "#4521"  -> "number 4521"
    (re.compile(r"#(?=[A-Za-z])"), ""),       # "#ORD-24589" -> "ORD-24589"
    (_TIME, _say_time),                       # "3:15 PM" -> "three fifteen PM"
]


def normalize_for_speech(text):
    for pattern, replacement in RULES:
        text = pattern.sub(replacement, text)
    return text
