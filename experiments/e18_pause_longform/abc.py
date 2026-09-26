#!/usr/bin/env python3
"""A/B/C: hand-written table vs e16 pause model vs e18 (long-form) pause model.

Same voice, speed, seed, and piece cuts; the ONLY difference is pause placement.
  A  table: PAUSES_MS ranges per punctuation
  B  e16:   trained on LibriSpeech subset (mostly punctuation->pause)
  C  e18:   trained on 5.6h LibriVox audiobooks (same 93k-param BiLSTM)

Usage:
    ~/workspace/.pause-venv/bin/python abc.py
    # writes wav/01_list_a_table.wav, wav/01_list_b_e16.wav, wav/01_list_c_e18.wav, ...
"""
import sys
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "e16_pause_model"))
sys.path.insert(0, str(REPO / "experiments" / "e18_pause_longform"))

from toast.pacing import speak_naturally  # noqa: E402
from toast.voices import load_voice  # noqa: E402
from toast.learned_pauses import learned_pauses  # noqa: E402
from predict import PausePredictor  # noqa: E402

VOICE = "amy"
SPEED = 1.2
SEED = 7

TEXTS = {
    "01_list": "She bought apples, oranges, and bananas. Then she left.",
    "02_opener": "Well, to be honest, I wasn't expecting that. But here we are.",
    "03_flow": "Although it was raining, we went outside, and we stayed for hours.",
    "04_semicolon": "Does it handle micro-breaks, like after a semicolon; or does it just rush on?",
    # No punctuation at all: does any model insert phrase breaks?
    "05_nopunct": "She bought apples oranges and bananas then she left the store",
    # Colon: absent from e16's training; e18 audiobooks have them.
    "06_colon": "He had one goal: to finish the race before sunset.",
}


def write_wav(path, audio, sr):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main():
    outdir = Path(__file__).resolve().parent / "wav"
    outdir.mkdir(exist_ok=True)
    voice = load_voice(VOICE)
    pred16 = PausePredictor(str(REPO / "experiments" / "e16_pause_model" / "data" / "model" / "pause_model.pt"))
    pred18 = PausePredictor(str(REPO / "experiments" / "e18_pause_longform" / "data" / "model" / "pause_model.pt"))

    for name, text in TEXTS.items():
        print(f"--- {name}: {text}")
        audio_a = speak_naturally(voice, text, speed=SPEED, seed=SEED)
        audio_b = speak_naturally(voice, text, speed=SPEED, seed=SEED,
                                  pause_ms=learned_pauses(pred16, text))
        audio_c = speak_naturally(voice, text, speed=SPEED, seed=SEED,
                                  pause_ms=learned_pauses(pred18, text))
        for tag, audio in (("a_table", audio_a), ("b_e16", audio_b), ("c_e18", audio_c)):
            p = outdir / f"{name}_{tag}.wav"
            write_wav(p, audio, voice.sample_rate)
        print(f"    A {len(audio_a)/voice.sample_rate:.2f}s | "
              f"B {len(audio_b)/voice.sample_rate:.2f}s | "
              f"C {len(audio_c)/voice.sample_rate:.2f}s")

    print(f"\ndone -> {outdir}")


if __name__ == "__main__":
    main()
