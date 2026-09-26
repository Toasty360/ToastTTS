"""E12: How do voices say the opening "Well,"?

Listening note: "the way they say 'Well,' is totally different." In our
pipeline "Well," is a one-word piece (smart split cuts after short pieces),
spoken in isolation and then softened. This compares the word itself:
duration, loudness shape and pitch shape, located with Whisper word
timestamps, across the cloud references, Kokoro, and amy in four ways:
  - pipeline:       as ToastTTS speaks it now (isolated piece + softening)
  - isolated:       the model given just "Well," (no softening)
  - in context:     the model given the whole first sentence, "Well," cut out
  - slow isolated:  "Well," alone at 0.8x speed

  .venv\\Scripts\\python experiments\\e12_well.py
"""

from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf

from toast.pacing import speak_naturally, soften_ending, trim_silence
from toast.speech_stats import timed_words
from toast.voices import load_voice

ROOT = Path(__file__).resolve().parents[1]
FIRST_SENTENCE = ("Well, to be honest, building an ultra-low latency voice pipeline isn't just about raw speed; "
                  "it's about cadence, rhythm, and natural breathing.")


def describe_word(audio, sr, start, end):
    seg = audio[int(start * sr):int(end * sr)].astype(np.float64)
    n = len(seg) // 4
    loud = [20 * np.log10(np.sqrt(np.mean(seg[i * n:(i + 1) * n] ** 2)) + 1e-9) for i in range(4)]
    pitch = parselmouth.Sound(seg, sampling_frequency=sr).to_pitch_ac(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
    hz = pitch.selected_array["frequency"]
    hz = hz[hz > 0]
    if len(hz) >= 3:
        st = 12 * np.log2(hz / np.median(hz))
        third = max(1, len(st) // 3)
        shape = f"start {np.mean(st[:third]):+.1f} / mid {np.mean(st[third:-third] if len(st) > 2 else st):+.1f} / end {np.mean(st[-third:]):+.1f} st, range {st.max() - st.min():.1f} st"
    else:
        shape = "pitch not found"
    return f"{(end - start) * 1000:4.0f} ms | loudness by quarter (dB, rel. to 1st): " \
           f"{' '.join(f'{x - loud[0]:+.1f}' for x in loud)} | pitch {shape}"


def first_word(audio, sr):
    words = timed_words(audio, sr)
    return words[0] if words else ("?", 0.0, len(audio) / sr)


def main():
    references = {
        "Soniox Grace": ROOT / "cloud_voices" / "soniox" / "soniox-tts-grace.wav",
        "Deepgram Thalia": ROOT / "cloud_voices" / "deepgram" / "deepgram-aura-2-thalia-en.wav",
        "Kokoro af_heart": ROOT / "out" / "listen" / "21_kokoro_af_heart_toast_1.0x.wav",
    }
    for label, path in references.items():
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        word, start, end = first_word(audio, sr)
        print(f"{label:<22} {word!r:<8} {describe_word(audio, sr, start, end)}")

    amy = load_voice("amy")
    sr = amy.sample_rate
    variants = {
        "amy pipeline": speak_naturally(amy, FIRST_SENTENCE, speed=1.2, seed=0),
        "amy isolated": trim_silence(amy.synthesize("Well,", speed=1.2), sr),
        "amy in context": amy.synthesize(FIRST_SENTENCE, speed=1.2),
        "amy slow isolated": trim_silence(amy.synthesize("Well,", speed=0.8), sr),
    }
    for label, audio in variants.items():
        word, start, end = first_word(audio, sr)
        print(f"{label:<22} {word!r:<8} {describe_word(audio, sr, start, end)}")
        sf.write(ROOT / "out" / "listen" / f"50_well_{label.replace(' ', '_')}.wav", audio[: int((end + 0.3) * sr)], sr)


if __name__ == "__main__":
    main()
