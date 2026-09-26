"""Measure how a recording is paced: speaking rate and pauses.

Whisper gives each word a start and end time. From those we get:
  - words per minute, counting only time spent talking (pauses excluded)
    and overall (pauses included)
  - every pause between words, grouped by the punctuation before it
    ("after ,", "after .", "none" = a pause mid-phrase)

Used to compare our output with cloud voices and tune toward them.
"""

import numpy as np

from toast.word_check import _whisper

MIN_PAUSE_S = 0.08  # shorter gaps are just the space between words


def _to_16k(audio, sample_rate):
    times = np.linspace(0, len(audio) / sample_rate, int(len(audio) * 16000 / sample_rate), endpoint=False)
    return np.interp(times, np.arange(len(audio)) / sample_rate, audio).astype(np.float32)


def timed_words(audio, sample_rate):
    segments, _ = _whisper().transcribe(_to_16k(audio, sample_rate), language="en", beam_size=5,
                                        word_timestamps=True, condition_on_previous_text=False)
    return [(w.word.strip(), w.start, w.end) for s in segments for w in s.words]


def pause_kind(word):
    if word.endswith(("...", "…")):
        return "..."
    if word and word[-1] in ",;:.?!—-":
        return {"-": "—"}.get(word[-1], word[-1])
    return "none"


def speech_stats(audio, sample_rate):
    words = timed_words(audio, sample_rate)
    pauses = {}
    for (word, _, end), (_, next_start, _) in zip(words, words[1:]):
        gap = next_start - end
        if gap >= MIN_PAUSE_S or pause_kind(word) != "none":
            pauses.setdefault(pause_kind(word), []).append(gap)
    spoken = words[-1][2] - words[0][1]
    all_pauses = sum(g for gaps in pauses.values() for g in gaps if g >= MIN_PAUSE_S)
    return {
        "text": " ".join(w for w, _, _ in words),
        "words": len(words),
        "seconds": len(audio) / sample_rate,
        "wpm_overall": 60 * len(words) / spoken,
        "wpm_talking": 60 * len(words) / max(spoken - all_pauses, 1e-6),
        "pauses": pauses,
    }


def describe(stats):
    lines = [f"  {stats['words']} words, {stats['seconds']:.1f}s, "
             f"{stats['wpm_overall']:.0f} wpm overall, {stats['wpm_talking']:.0f} wpm while talking"]
    for kind in [",", ".", "?", ";", ":", "—", "...", "!", "none"]:
        gaps = [g for g in stats["pauses"].get(kind, []) if kind != "none" or g >= MIN_PAUSE_S]
        if gaps:
            lines.append(f"    pause after {kind!r:<6} n={len(gaps):<3} median {1000 * np.median(gaps):4.0f} ms"
                         f"  (range {1000 * min(gaps):.0f}-{1000 * max(gaps):.0f})")
    return "\n".join(lines)
