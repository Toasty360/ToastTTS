#!/usr/bin/env python3
"""A/B: current PAUSES_MS table vs the learned e16 pause model.

Renders each sample text twice with the same voice, speed, seed, and piece
cuts. The ONLY difference is how long each pause lasts:

  A (current)  pause from PAUSES_MS: random in a hand-set range per
               punctuation (comma 180-260 ms, period 480-620 ms, ...)
  B (learned)  pause from the e16 model: 440 ms after commas, ~720 ms after
               sentence-final stops, 0 ms where it predicts no pause

Same pieces, same voice, same seed otherwise. Listen to each pair and pick
the more natural one; ears decide, not metrics.

Usage:
    ~/workspace/.pause-venv/bin/python ab.py
    # writes wav/01_list_a_table.wav, wav/01_list_b_learned.wav, ...

Requires: piper-tts in the venv, models/en_US-amy-medium.onnx in the repo,
and experiments/e16_pause_model/data/model/pause_model.pt.
"""
import sys
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "e16_pause_model"))

from toast.pacing import speak_naturally  # noqa: E402
from toast.voices import load_voice  # noqa: E402
from toast.learned_pauses import learned_pauses  # noqa: E402
from predict import PausePredictor  # noqa: E402

VOICE = "amy"
SPEED = 1.2  # amy's usual rate in the benchmarks
SEED = 7

TEXTS = {
    # The model's home turf: list commas + a period.
    "01_list": "She bought apples, oranges, and bananas. Then she left.",
    # Opener rule (60-140 ms) vs the model's flat 440 ms comma.
    "02_opener": "Well, to be honest, I wasn't expecting that. But here we are.",
    # Subordinate-clause commas, no list.
    "03_flow": "Although it was raining, we went outside, and we stayed for hours.",
    # The honest gap: the model never saw a semicolon, so it predicts no
    # pause there; the table gives 300-380 ms.
    "04_semicolon": "Does it handle micro-breaks, like after a semicolon; or does it just rush on?",
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
    predictor = PausePredictor(str(REPO / "experiments" / "e16_pause_model" / "data" / "model" / "pause_model.pt"))

    for name, text in TEXTS.items():
        print(f"--- {name}: {text}")
        # A: current table pauses.
        audio_a = speak_naturally(voice, text, speed=SPEED, seed=SEED)
        # B: learned pauses (same pieces, same seed otherwise).
        audio_b = speak_naturally(voice, text, speed=SPEED, seed=SEED,
                                  pause_ms=learned_pauses(predictor, text))
        pa, pb = outdir / f"{name}_a_table.wav", outdir / f"{name}_b_learned.wav"
        write_wav(pa, audio_a, voice.sample_rate)
        write_wav(pb, audio_b, voice.sample_rate)
        print(f"    A {pa.name} {len(audio_a) / voice.sample_rate:.1f}s | "
              f"B {pb.name} {len(audio_b) / voice.sample_rate:.1f}s")

    print(f"\ndone -> {outdir}")


if __name__ == "__main__":
    main()
