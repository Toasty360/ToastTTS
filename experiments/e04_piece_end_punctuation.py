"""E04: How does the punctuation at the end of a piece change the last word?

Listening note: "semicolon;" at the end of a piece sounded swallowed.
Hypothesis: Piper rushes the final word when a piece ends in ";" but
finishes it fully before "," (which signals "more is coming").

Method: synthesize the same piece with different endings, 6 runs each
(Piper output is random per run), compare durations and loudness of the
final 250 ms.

Finding: ";" gives the shortest audio; "," the longest, with the word
still loud near the end. So pieces ending mid-sentence in ; : or a dash
are sent to the model with "," (toast/pacing.py _text_for_model).
(The *start* of "semicolon" being swallowed turned out to be a separate,
voice-level problem: see E05.)

  .venv\\Scripts\\python experiments\\e04_piece_end_punctuation.py
"""

import numpy as np

from toast.voices import PiperEngine

BASE = "Does it handle subtle micro-breaks, like after a semicolon"


def main():
    voice = PiperEngine("en_US-libritts_r-medium", "3922", noise_scale=0.5, noise_w_scale=0.8)
    sr = voice.sample_rate
    for ending in [";", ",", ".", ""]:
        lengths, tails = [], []
        for _ in range(6):
            audio = voice.synthesize(BASE + ending, speed=0.9)
            n = int(sr * 0.01)
            usable = len(audio) // n * n
            db = 20 * np.log10(np.sqrt(np.mean(audio[:usable].reshape(-1, n) ** 2, axis=1))
                               / np.abs(audio).max() + 1e-12)
            lengths.append(len(audio) / sr)
            tails.append(float(np.max(db[-25:])))
        print(f"  ending {ending!r:<5} length {np.mean(lengths):.2f}s (+/- {np.std(lengths):.2f})   "
              f"loudest point in last 250 ms: {np.mean(tails):6.1f} dB")


if __name__ == "__main__":
    main()
