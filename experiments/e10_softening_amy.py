"""E10: Give amy a soft last word before each pause, and calibrate it.

E09 showed natural voices (Kokoro, Soniox) get quieter on the word before a
pause while amy stays at full volume. toast/pacing.py soften_ending() eases
the volume down over the last SOFT_TAIL_MS of every piece by SOFT_TAIL_DB.
This sweeps those two settings, measures each with E09's method (2 renders
each), and writes listening files with identical pauses (same seed), so
softening is the only difference between them.

  .venv\\Scripts\\python experiments\\e10_softening_amy.py
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e09_phrase_final_softening import final_word_profile  # noqa: E402

from toast import pacing  # noqa: E402
from toast.voices import load_voice  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = [(0, 0.0), (300, -4.0), (400, -4.0), (400, -6.0), (500, -6.0), (500, -9.0)]
LISTEN = {(0, 0.0): "30_amy_1.2x_no_softening", (400, -4.0): "31_amy_1.2x_soft_400ms_-4dB",
          (500, -6.0): "32_amy_1.2x_soft_500ms_-6dB_default", (500, -9.0): "33_amy_1.2x_soft_500ms_-9dB"}


def main():
    text = (ROOT / "samples" / "reference.txt").read_text(encoding="utf-8-sig").strip()
    amy = load_voice("amy")
    print("Targets (E09): Kokoro drop -0.7 / fade -3.8; Soniox drop -0.3 / fade -3.2")
    print(f"{'tail ms':>8}{'depth dB':>9}{'drop':>7}{'fade':>7}{'n':>4}   listening file")
    for tail, depth in SETTINGS:
        pacing.SOFT_TAIL_MS, pacing.SOFT_TAIL_DB = tail, depth
        drops, fades = [], []
        for seed in (0, 1):
            audio = pacing.speak_naturally(amy, text, speed=1.2, seed=seed)
            d, f = final_word_profile(audio, amy.sample_rate)
            drops += list(d)
            fades += list(f)
            if seed == 0 and (tail, depth) in LISTEN:
                path = ROOT / "out" / "listen" / f"{LISTEN[(tail, depth)]}.wav"
                sf.write(path, audio, amy.sample_rate)
        name = LISTEN.get((tail, depth), "")
        print(f"{tail:>8}{depth:>9.1f}{np.median(drops):>7.1f}{np.median(fades):>7.1f}{len(drops):>4}   {name}")


if __name__ == "__main__":
    main()
