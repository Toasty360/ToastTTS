"""E07: How much of our Coval word error rate is just how Whisper writes times?

In the Coval benchmark (scripts/bench_coval.py), Whisper writes spoken
times as "2.30 pm"; Coval's normalizer turns "2:30" into "2 30" but
leaves "2.30", so each time costs 2 word errors. This re-scores the saved
transcripts with only that formatting difference removed.

The adjusted number is for understanding our real mistakes, NOT for the
leaderboard comparison: we can't tell whether cloud providers were hit
the same way (E06 shows Whisper writes "3.15 pm" for Deepgram and Soniox
too, so probably yes).

  .venv\\Scripts\\python experiments\\e07_coval_time_format.py
"""

import csv
import re
from pathlib import Path

import jiwer

ROOT = Path(__file__).resolve().parents[1]
FILES = {"whisper-1 (Coval's method)": "coval_tts_v1_whisper1.csv", "local large-v2": "coval_tts_v1.csv"}


def without_time_format(text):
    text = re.sub(r"\b(\d{1,2})\.(\d\d)\b(?=\s*(am|pm|a m|p m))", r"\1 \2", text)
    return text.replace("p m", "pm").replace("a m", "am")


def main():
    for label, name in FILES.items():
        rows = list(csv.DictReader((ROOT / "benchmarks" / name).open(encoding="utf-8")))
        print(f"== scored by {label}")
        for voice in dict.fromkeys(r["voice"] for r in rows):
            mine = [r for r in rows if r["voice"] == voice]
            words = sum(int(r["ref_words"]) for r in mine)
            raw = sum(int(r["errors"]) for r in mine)
            fixed = 0
            for r in mine:
                m = jiwer.process_words(r["reference"], without_time_format(r["heard"]))
                fixed += m.substitutions + m.deletions + m.insertions
            print(f"  {voice:<12} WER {100 * raw / words:4.1f}%   without time-format errors "
                  f"{100 * fixed / words:4.1f}%   ({words} reference words)")

    rows = list(csv.DictReader((ROOT / "benchmarks" / FILES["whisper-1 (Coval's method)"]).open(encoding="utf-8")))
    print("\n== remaining real mistakes, amy (whisper-1), after removing time formatting")
    for r in rows:
        if r["voice"] != "amy":
            continue
        m = jiwer.process_words(r["reference"], without_time_format(r["heard"]))
        if m.substitutions + m.deletions + m.insertions:
            print(f"  {r['id']}/take{r['take']}: ref: {r['reference']}\n{'':15}heard: {without_time_format(r['heard'])}")


if __name__ == "__main__":
    main()
