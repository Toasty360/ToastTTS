"""E06: How do cloud voices pace speech, and how does amy compare?

Reference recordings (cloud_voices/, reading samples/reference.txt) from
Deepgram and Soniox. Measured with Whisper word timestamps
(toast/speech_stats.py): words per minute overall and while talking, and
pause lengths grouped by the punctuation before them. amy is rendered
through our full pipeline at several speeds and measured the same way.

Finding: our pause lengths already match the cloud voices (commas ~450 ms,
periods ~800 ms measured this way); amy's speaking rate was the gap
(152 wpm at 0.9x vs 183-195 for cloud). 1.2x reaches ~180 wpm, so it
became the default. Whisper also writes "3.15 pm" for cloud voices.

  .venv\\Scripts\\python experiments\\e06_cloud_pacing.py
"""

from pathlib import Path

import soundfile as sf

from toast.naturalness import naturalness
from toast.pacing import speak_naturally
from toast.speech_stats import describe, speech_stats
from toast.voices import load_voice

ROOT = Path(__file__).resolve().parents[1]


def report(label, audio, sample_rate):
    stats = speech_stats(audio, sample_rate)
    print(f"== {label}   natural (first 20 s): {naturalness(audio[: sample_rate * 20], sample_rate):.2f}")
    print(describe(stats))


def main():
    for path in sorted((ROOT / "cloud_voices").rglob("*.wav")):
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        report(f"cloud: {path.name}", audio.mean(axis=1), sr)

    text = (ROOT / "samples" / "reference.txt").read_text(encoding="utf-8-sig").strip()
    voice = load_voice("amy")
    for speed in (0.9, 1.0, 1.1, 1.2):
        report(f"ToastTTS amy x{speed}", speak_naturally(voice, text, speed=speed), voice.sample_rate)


if __name__ == "__main__":
    main()
