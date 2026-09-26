"""E02: Do the models already pause at punctuation?

Idea being tested: find the model's own pauses (quiet stretches) and
stretch them, instead of cutting the text. For that to work, the model
must leave clear gaps at commas and few gaps elsewhere.

Method: synthesize sentences with many commas; mark 10 ms frames quieter
than -40 dB (relative to the peak); list internal quiet runs >= 30 ms.

Finding (see research/notebook/02-natural-pauses.md): Piper libritts_r 3922
barely pauses at commas (a few 40-60 ms gaps for 7 punctuation marks);
Kitten leaves gaps between ordinary words too. So pauses can't be found
reliably in the audio; we insert our own between pieces instead.

  .venv\\Scripts\\python experiments\\e02_model_pause_gaps.py
"""

import numpy as np

from toast.voices import load_voice

SENTENCES = [
    "Well, to be honest, building an ultra-low latency voice pipeline isn't just about raw speed; "
    "it's about cadence, rhythm, and natural breathing.",
    "Take a look at these numbers: 12, 45, and 108.",
    "If the engine speaks at 3:15 PM without any dynamic pauses",
]


def quiet_runs(audio, sample_rate, threshold_db=-40, min_ms=30):
    frame = int(sample_rate * 0.01)
    usable = len(audio) // frame * frame
    rms = np.sqrt(np.mean(audio[:usable].reshape(-1, frame) ** 2, axis=1) + 1e-12)
    quiet = 20 * np.log10(rms / (np.abs(audio).max() + 1e-12)) < threshold_db
    runs, start = [], None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        if not q and start is not None:
            runs.append((start, i))
            start = None
    return [(s * 10, (e - s) * 10) for s, e in runs if s > 0 and (e - s) * 10 >= min_ms]


def main():
    for name in ["3922", "kitten:micro", "amy"]:
        voice = load_voice(name)
        print(f"== {voice.name}")
        for sentence in SENTENCES:
            audio = voice.synthesize(sentence)
            marks = sum(sentence.count(c) for c in ",;:")
            gaps = quiet_runs(audio, voice.sample_rate)
            print(f"  {len(audio) / voice.sample_rate:4.1f}s, {marks} commas/;/:, {len(gaps):2} gaps  "
                  f"(at ms, length ms): {gaps}")
            print(f"     {sentence[:60]}")


if __name__ == "__main__":
    main()
