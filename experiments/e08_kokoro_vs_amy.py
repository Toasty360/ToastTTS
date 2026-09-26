"""E08: Does Kokoro close the prosody gap, and can it stream fast enough?

Kokoro-82M is a larger, more expressive open model (Apache-2.0). Piper amy
is our fast default. Questions:
  1. Naturalness and pacing: through our pipeline, is Kokoro closer to the
     cloud reference recordings than amy? (UTMOS + Whisper pacing, as in E06)
  2. Streaming: TTFA from the first LLM token and stalls, at 30 and 10
     tokens/s. Kokoro is only ~3-5x real time on this CPU, so stalls are a risk.
  3. Listening: renders go to out/listen/2x_*.wav, including Kokoro on its
     own (its built-in pauses, whole paragraph) for comparison.

  .venv\\Scripts\\python experiments\\e08_kokoro_vs_amy.py
  .venv\\Scripts\\python experiments\\e08_kokoro_vs_amy.py --kokoro kokoro:af_bella
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from toast.fake_llm import fake_llm_tokens
from toast.metrics import measure
from toast.naturalness import naturalness
from toast.pacing import speak_naturally, stream_from_llm
from toast.speech_stats import describe, speech_stats
from toast.voices import load_voice

ROOT = Path(__file__).resolve().parents[1]
LISTEN = ROOT / "out" / "listen"


def report(label, audio, sample_rate, path=None):
    if path:
        sf.write(path, audio, sample_rate)
    stats = speech_stats(audio, sample_rate)
    print(f"== {label}   natural (first 20 s): {naturalness(audio[: sample_rate * 20], sample_rate):.2f}"
          + (f"   -> {path.relative_to(ROOT)}" if path else ""))
    print(describe(stats))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kokoro", default="kokoro:af_heart")
    args = parser.parse_args()
    text = (ROOT / "samples" / "reference.txt").read_text(encoding="utf-8-sig").strip()
    LISTEN.mkdir(parents=True, exist_ok=True)
    tag = args.kokoro.replace(":", "_")

    print("# 1. Naturalness and pacing (reference: Soniox Grace 195 wpm talking, commas ~440 ms, "
          "periods ~800 ms, natural 4.47; see E06)")
    kokoro = load_voice(args.kokoro)
    kokoro.synthesize("Warm up.")
    raw, sr = kokoro._kokoro.create(text, voice=kokoro._voice, lang=kokoro._lang)  # Kokoro's own pauses
    report(f"{kokoro.name}, whole paragraph, Kokoro's own pacing", np.asarray(raw, dtype=np.float32), sr,
           LISTEN / f"20_{tag}_own_pacing.wav")
    for speed in (1.0, 1.1):
        report(f"{kokoro.name} x{speed} through ToastTTS", speak_naturally(kokoro, text, speed=speed, seed=0),
               kokoro.sample_rate, LISTEN / f"21_{tag}_toast_{speed}x.wav")
    amy = load_voice("amy")
    report("amy x1.2 through ToastTTS (current default)", speak_naturally(amy, text, speed=1.2, seed=0),
           amy.sample_rate, LISTEN / "16_amy_1.2x.wav")

    print("\n# 2. Streaming from a fake LLM (TTFA from first token, median of 3; stalls = worst run)")
    for voice, speed in ((kokoro, 1.0), (amy, 1.2)):
        for tps in (30, 10):
            def chunks():
                return stream_from_llm(voice, fake_llm_tokens(text, tps), speed=speed)

            _, stats = measure(chunks, voice.sample_rate, repeats=3)
            print(f"  {voice.name:<32} x{speed}  {tps:>2} tokens/s: TTFA {stats['ttfa_ms']:5.0f} ms, "
                  f"stalls {stats['stalls']} ({stats['stall_ms']:.0f} ms), made {stats['x_realtime']:.1f}x real time")


if __name__ == "__main__":
    main()
