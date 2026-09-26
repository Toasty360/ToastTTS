"""E11: Is amy "robotic" because its pitch moves too little?

Listening note: amy sounds robotic next to Kokoro af_heart. Hypothesis:
the difference is the melody (prosody), not the voice. Method:
  1. Pitch profile (toast/prosody.py) of Soniox Grace, Deepgram Thalia,
     Kokoro af_heart and amy, all reading samples/reference.txt.
  2. Scale amy's pitch movement (Praat PSOLA) to match af_heart's and
     Grace's spread, and write listening files.
  3. Check the edit didn't damage the audio: naturalness (UTMOS) and the
     word check before/after.

  .venv\\Scripts\\python experiments\\e11_pitch_profile.py
"""

from pathlib import Path

import soundfile as sf

from toast.naturalness import naturalness
from toast.pacing import speak_naturally
from toast.prosody import expand_pitch, pitch_profile
from toast.voices import load_voice
from toast.word_check import transcribe, word_errors

ROOT = Path(__file__).resolve().parents[1]
LISTEN = ROOT / "out" / "listen"
REFERENCES = {
    "Soniox Grace": ROOT / "cloud_voices" / "soniox" / "soniox-tts-grace.wav",
    "Deepgram Thalia": ROOT / "cloud_voices" / "deepgram" / "deepgram-aura-2-thalia-en.wav",
    "Kokoro af_heart (ToastTTS 1.0x)": LISTEN / "21_kokoro_af_heart_toast_1.0x.wav",
}


def row(label, profile, extra=""):
    print(f"  {label:<38}{profile['median_hz']:>7.0f}{profile['spread_st']:>8.2f}{profile['range_st']:>8.2f}"
          f"{profile['movement_st_s']:>10.1f}  {extra}")


def main():
    text = (ROOT / "samples" / "reference.txt").read_text(encoding="utf-8-sig").strip()
    print(f"  {'recording':<38}{'Hz':>7}{'spread':>8}{'range':>8}{'movement':>10}")
    print(f"  {'':<38}{'median':>7}{'st':>8}{'st':>8}{'st/s':>10}")
    profiles = {}
    for label, path in REFERENCES.items():
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        profiles[label] = pitch_profile(audio.mean(axis=1), sr)
        row(label, profiles[label])

    amy = load_voice("amy")
    sr = amy.sample_rate
    base = speak_naturally(amy, text, speed=1.2, seed=0)
    amy_profile = pitch_profile(base, sr)
    row("amy 1.2x (current default)", amy_profile)

    print("\nScaling amy's pitch movement to match each reference's spread:")
    outputs = {"as is": (1.0, base)}
    for label, name in (("Kokoro af_heart (ToastTTS 1.0x)", "af_heart"), ("Soniox Grace", "grace")):
        factor = profiles[label]["spread_st"] / amy_profile["spread_st"]
        edited = expand_pitch(base, sr, factor)
        path = LISTEN / f"40_amy_pitch_like_{name}_x{factor:.2f}.wav"
        sf.write(path, edited, sr)
        outputs[f"like {name}"] = (factor, edited)
        row(f"amy, pitch x{factor:.2f} (like {name})", pitch_profile(edited, sr), f"-> {path.name}")
    sf.write(LISTEN / "40_amy_pitch_as_is.wav", base, sr)

    print("\nDid the edit damage the audio? (naturalness on first 20 s; wrong words of 95)")
    for label, (factor, audio) in outputs.items():
        wrong, _ = word_errors(text, transcribe(audio, sr))
        print(f"  amy {label:<16} x{factor:.2f}: natural {naturalness(audio[: sr * 20], sr):.2f}, wrong words {wrong}")


if __name__ == "__main__":
    main()
