"""E03: Does turning up Piper's randomness cause the "gasping" sound?

Listening note: 3922 with "lively" settings (noise_scale 0.667, noise_w
0.8) had gasps/breath artifacts. Hypothesis: noise_scale (randomness of
the sound itself) brings out breath noise the voice learned from
audiobook recordings; noise_w (randomness of timing) does not.

Method: count "breathy" 10 ms frames: noise-like (spectral flatness
> 0.35) yet audible (-38 to -12 dB relative to peak). Caveat: "s", "f",
"sh" also count, so compare settings against each other, not absolutely.

  .venv\\Scripts\\python experiments\\e03_randomness_breathiness.py
"""

from pathlib import Path

import numpy as np

from toast.splitter import split_into_sentences
from toast.voices import PiperEngine

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {
    "file defaults 0.333/0.333": (None, None),
    "rhythm only 0.333/0.8 (4b)": (0.333, 0.8),
    "halfway 0.5/0.8 (4c)": (0.5, 0.8),
    "lively 0.667/0.8": (0.667, 0.8),
}


def breathy_frames(audio, sample_rate):
    n = int(sample_rate * 0.01)
    usable = len(audio) // n * n
    frames = audio[:usable].reshape(-1, n) * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(frames, axis=1)) + 1e-9
    flatness = np.exp(np.mean(np.log(spectrum), axis=1)) / np.mean(spectrum, axis=1)
    db = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) / (np.abs(audio).max() + 1e-9) + 1e-12)
    return int(np.sum((flatness > 0.35) & (db > -38) & (db < -12)))


def main():
    text = (ROOT / "samples" / "reference.txt").read_text(encoding="utf-8-sig")
    sentences = split_into_sentences(text)
    print("Breathy frames per paragraph (sum over 5 sentences, mean of 3 runs; speed 0.9)")
    for label, (noise, noise_w) in SETTINGS.items():
        voice = PiperEngine("en_US-libritts_r-medium", "3922", noise_scale=noise, noise_w_scale=noise_w)
        totals = [sum(breathy_frames(voice.synthesize(s, speed=0.9), voice.sample_rate) for s in sentences)
                  for _ in range(3)]
        print(f"  {label:<28} {np.mean(totals):6.1f}  (runs: {totals})")


if __name__ == "__main__":
    main()
