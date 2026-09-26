"""E09: How much quieter is the last word before a pause?

Listening note: Kokoro sounds natural largely because "the last word's
amplitude before a natural pause is soft"; amy's is not. This measures it.

Method: Whisper word timestamps locate every word followed by a pause
(gap >= 150 ms). For each, compare loudness (RMS, dB):
  - drop     = last word vs the speech in the 1.5 s before it (how much
               quieter the final word is overall)
  - fade     = last 30% of the last word vs its first 70% (how much the
               word itself decays toward the pause)
Medians over all pauses in a recording of samples/reference.txt.

  .venv\\Scripts\\python experiments\\e09_phrase_final_softening.py
  .venv\\Scripts\\python experiments\\e09_phrase_final_softening.py out\\listen\\some.wav ...
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from toast.speech_stats import timed_words

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = [
    ROOT / "cloud_voices" / "soniox" / "soniox-tts-grace.wav",
    ROOT / "cloud_voices" / "deepgram" / "deepgram-aura-2-thalia-en.wav",
    ROOT / "out" / "listen" / "20_kokoro_af_heart_own_pacing.wav",
    ROOT / "out" / "listen" / "21_kokoro_af_heart_toast_1.0x.wav",
    ROOT / "out" / "listen" / "16_amy_1.2x.wav",
]


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)


def final_word_profile(audio, sr, min_gap=0.15):
    words = timed_words(audio, sr)
    drops, fades = [], []
    for (word, start, end), (_, next_start, _) in zip(words, words[1:]):
        if next_start - end < min_gap or end - start < 0.12:
            continue
        a, b = int(start * sr), int(end * sr)
        before = audio[max(0, a - int(1.5 * sr)):a]
        last = audio[a:b]
        split = a + int(0.7 * (b - a))
        if len(before) < sr * 0.3:
            continue
        drops.append(rms_db(last) - rms_db(before))
        fades.append(rms_db(audio[split:b]) - rms_db(audio[a:split]))
    return np.array(drops), np.array(fades)


def main():
    files = [Path(p) for p in sys.argv[1:]] or DEFAULT_FILES
    print(f"{'recording':<48}{'pauses':>7}{'drop dB':>9}{'fade dB':>9}")
    for path in files:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        drops, fades = final_word_profile(audio.mean(axis=1), sr)
        print(f"{path.name:<48}{len(drops):>7}{np.median(drops):>9.1f}{np.median(fades):>9.1f}")
    print("\n drop = last word before a pause vs the 1.5 s before it (negative = softer)")
    print(" fade = last 30% of that word vs its first 70% (negative = decays into the pause)")


if __name__ == "__main__":
    main()
