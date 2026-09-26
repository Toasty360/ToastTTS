"""Hear word emphasis change a sentence's meaning.

Renders the same sentence with the stress on a different word each time, plus a
plain version. Checks that every word stays intelligible (Whisper) and measures
how much higher the marked word's pitch is than in the plain version (Praat).

  .venv\\Scripts\\python scripts\\emphasis_demo.py
  .venv\\Scripts\\python scripts\\emphasis_demo.py "I never said she stole my money."
  .venv\\Scripts\\python scripts\\emphasis_demo.py --strong        # bigger accent + post-focus compression
"""

import sys
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf
from piper import PiperVoice, SynthesisConfig

from toast.emphasis import STRENGTHS, parse, speak_with_emphasis, word_spans
from toast.word_check import transcribe, word_errors

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "aligned" / "en_US-amy-medium.onnx"
OUT = ROOT / "out" / "listen" / "emphasis"
LENGTH_SCALE = 1 / 1.2


def median_pitch(audio, sr, start, end):
    segment = audio[int(start):int(end)].astype(np.float64)
    hz = parselmouth.Sound(segment, sampling_frequency=sr).to_pitch_ac(
        time_step=0.005, pitch_floor=75, pitch_ceiling=600).selected_array["frequency"]
    hz = hz[hz > 0]
    return float(np.median(hz)) if len(hz) else float("nan")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    strength = "strong" if "--strong" in sys.argv else "subtle"
    lengthen = STRENGTHS[strength][1]
    prefix = "strong_" if strength == "strong" else ""
    sentence = args[0] if args else "I didn't say he stole it."
    OUT.mkdir(parents=True, exist_ok=True)
    voice = PiperVoice.load(MODEL)
    sr = voice.config.sample_rate
    words = sentence.split()

    plain_audio, _, _ = speak_with_emphasis(voice, sentence, LENGTH_SCALE)
    sf.write(OUT / "plain.wav", plain_audio, sr)
    spans = []
    for chunk in voice.synthesize(sentence, SynthesisConfig(length_scale=LENGTH_SCALE), include_alignments=True):
        spans += word_spans(chunk.phoneme_alignments or [], sr)

    print(f"{'file':<24}{'wrong':>6}{'marked word pitch vs plain':>28}  text")
    print(f"{'plain.wav':<24}{word_errors(sentence, transcribe(plain_audio, sr))[0]:>6}{'':>28}  {sentence}")
    for i, word in enumerate(words):
        text = " ".join(f"*{x}*" if j == i else x for j, x in enumerate(words))
        audio, _, info = speak_with_emphasis(voice, text, LENGTH_SCALE, strength)
        name = f"{prefix}stress_{i + 1}_{word.strip('.,?!').lower()}.wav"
        sf.write(OUT / name, audio, sr)
        wrong = word_errors(parse(text)[0], transcribe(audio, sr))[0]
        start, end = spans[i]  # words before the marked one are unchanged, so it starts in the same place
        plain_hz = median_pitch(plain_audio, sr, start, end)
        stressed_hz = median_pitch(audio, sr, start, start + (end - start) * lengthen)
        rise = 12 * np.log2(stressed_hz / plain_hz)
        print(f"{name:<24}{wrong:>6}{rise:>+24.1f} st  {text}")
    print(f"\nListen: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
