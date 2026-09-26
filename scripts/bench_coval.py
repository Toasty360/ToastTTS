"""Run Coval's public TTS benchmark (tts-v1) on our engine, their way.

Coval (benchmarks.coval.ai/tts) ranks cloud TTS providers such as
ElevenLabs, OpenAI, Deepgram and Cartesia. Their method is open source
(github.com/coval-ai/benchmarks), so we copy it as closely as we can:

  prompts     their 30 customer-service sentences (samples/coval_tts_v1.json,
              Apache-2.0): order numbers, prices, times, tracking codes
  WER         audio transcribed by OpenAI's hosted "whisper-1" like Coval
              (--asr whisper-1, needs OPENAI_API_KEY, ~$0.17 per full run),
              or by Whisper large-v2 on this laptop (--asr local, free, slow),
              both texts cleaned with whisper_normalizer's
              EnglishTextNormalizer, errors pooled over all clips
              (sum of mistakes / sum of reference words)
  TTFA        time until the first chunk + silence before the first audible
              sound (their RMS > 0.01 rule on 10 ms frames), P50 over clips

One honest difference: cloud providers' TTFA includes the network round
trip; ours runs on this laptop, so there is none. That's the point of an
on-device engine, but it's not a like-for-like race.

  .venv\\Scripts\\python scripts\\bench_coval.py
  .venv\\Scripts\\python scripts\\bench_coval.py --asr whisper-1
  .venv\\Scripts\\python scripts\\bench_coval.py --voices lessac ryan --takes 1
"""

import argparse
import csv
import io
import json
import os
import time
from pathlib import Path

import jiwer
import numpy as np
from whisper_normalizer.english import EnglishTextNormalizer

from toast.pacing import stream_speech
from toast.voices import load_voice

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "samples" / "coval_tts_v1.json"
RESULTS_BY_ASR = {
    "local": ROOT / "benchmarks" / "coval_tts_v1.csv",
    "whisper-1": ROOT / "benchmarks" / "coval_tts_v1_whisper1.csv",
}
SPEED = 0.9

VOICES = {
    "lessac-high": "piper:en_US-lessac-high",
    "ryan": "piper:en_US-ryan-medium",
    "amy": "piper:en_US-amy-medium",
    "lessac": "piper:en_US-lessac-medium",
    "kitten": "kitten",
}

# Coval leaderboard, 1-day window read 2026-09-08 (as reprinted by
# gradium.ai/content/tts-wer-benchmark-2026; 14 of 26 models listed there).
COVAL_2026_09_08 = [
    ("Soniox TTS Rt v2", 4.0, 255), ("ElevenLabs Eleven v3 Conversational", 4.3, 320),
    ("Inworld TTS 2", 4.5, 170), ("Fish Audio S2.1 Pro", 4.7, 293), ("Gradium TTS", 4.9, 214),
    ("OpenAI GPT-4o mini TTS", 4.9, None), ("Deepgram Aura-2", 5.0, 290), ("Rime Mist v3", 5.0, 256),
    ("Fluxions vui", 5.3, 51), ("Inworld TTS Flash 2", 5.3, 75), ("Cartesia Sonic 3.6", 5.3, 440),
    ("Palabra TTS v1", 5.7, 103), ("Cartesia Sonic 3.5", 5.8, 269), ("ElevenLabs Flash v2.5", 6.5, 185),
]

_normalize = EnglishTextNormalizer()
_whisper = None


def first_audible_offset_ms(audio, sample_rate):
    """Coval's rule (metrics/ttfa.py, Apache-2.0): first 10 ms frame, stepped
    every 1 ms, whose RMS (DC removed) is above 0.01."""
    frame = max(1, round(0.010 * sample_rate))
    hop = max(1, round(0.001 * sample_rate))
    padded = np.pad(audio.astype(np.float32), frame // 2, mode="edge")
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame)[::hop]
    rms = np.sqrt(np.mean((frames - frames.mean(axis=1, keepdims=True)) ** 2, axis=1))
    audible = np.nonzero(rms > 0.01)[0]
    return None if audible.size == 0 else audible[0] * hop / sample_rate * 1000


def transcribe_large_v2(audio, sample_rate):
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel

        _whisper = WhisperModel("large-v2", device="cpu", compute_type="int8")
    times = np.linspace(0, len(audio) / sample_rate, int(len(audio) * 16000 / sample_rate), endpoint=False)
    audio16 = np.interp(times, np.arange(len(audio)) / sample_rate, audio).astype(np.float32)
    segments, _ = _whisper.transcribe(audio16, language="en", beam_size=5, condition_on_previous_text=False)
    return " ".join(s.text.strip() for s in segments)


def transcribe_whisper1(audio, sample_rate):
    """OpenAI's hosted whisper-1: exactly what Coval uses. $0.006 per minute."""
    global _openai
    if _openai is None:
        import openai

        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set. Run: setx OPENAI_API_KEY \"sk-...\" and restart.")
        _openai = openai.OpenAI()
    import soundfile as sf

    wav = io.BytesIO()
    sf.write(wav, audio, sample_rate, format="WAV", subtype="PCM_16")
    for attempt in range(3):
        try:
            response = _openai.audio.transcriptions.create(model="whisper-1", file=("clip.wav", wav.getvalue()))
            return response.text
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


_openai = None
TRANSCRIBERS = {"local": transcribe_large_v2, "whisper-1": transcribe_whisper1}


def synthesize_timed(voice, text, seed):
    """Our streaming pipeline with the full text available (like a TTS API call)."""
    start = time.perf_counter()
    chunks, ttfa = [], None
    for chunk in stream_speech(voice, text, speed=SPEED, seed=seed):
        if ttfa is None:
            arrived_ms = (time.perf_counter() - start) * 1000
            silence = first_audible_offset_ms(chunk, voice.sample_rate)
            ttfa = arrived_ms + (silence or 0.0)
        chunks.append(chunk)
    return np.concatenate(chunks), ttfa


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voices", nargs="*", default=["lessac-high", "ryan", "amy", "lessac"],
                        help=f"from: {', '.join(VOICES)}")
    parser.add_argument("--takes", type=int, default=2, help="times each prompt is spoken (voices vary per take)")
    parser.add_argument("--asr", choices=list(TRANSCRIBERS), default="local",
                        help="who listens: 'whisper-1' (OpenAI, what Coval uses) or 'local' (large-v2)")
    parser.add_argument("--ttfa-only", action="store_true",
                        help="re-time the saved results without Whisper, so all voices are timed under the "
                             "same conditions (e.g. same power mode); word errors are kept")
    args = parser.parse_args()

    items = json.loads(PROMPTS.read_text(encoding="utf-8"))["items"]
    results = RESULTS_BY_ASR[args.asr]
    if args.ttfa_only:
        retime(items, args.voices, args.takes, results)
        return
    transcribe = TRANSCRIBERS[args.asr]
    results.parent.mkdir(exist_ok=True)
    rows = []
    for short_name in args.voices:
        voice = load_voice(VOICES[short_name])
        voice.synthesize("Warm up.")
        started = time.perf_counter()
        for take in range(args.takes):
            for item in items:
                audio, ttfa = synthesize_timed(voice, item["transcript"], seed=take)
                heard = transcribe(audio, voice.sample_rate)
                reference, hypothesis = _normalize(item["transcript"]), _normalize(heard)
                counts = jiwer.process_words(reference, hypothesis)
                rows.append({
                    "voice": short_name, "id": item["testcase_id"], "take": take, "ttfa_ms": round(ttfa, 1),
                    "errors": counts.substitutions + counts.deletions + counts.insertions,
                    "ref_words": len(reference.split()), "reference": reference, "heard": hypothesis,
                })
        print(f"{short_name}: done in {time.perf_counter() - started:.0f}s", flush=True)

    with results.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print_table(rows, args.voices, results)


def retime(items, voices, takes, results):
    with results.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_key = {(r["voice"], r["id"], int(r["take"])): r for r in rows}
    for short_name in voices:
        voice = load_voice(VOICES[short_name])
        voice.synthesize("Warm up.")
        for take in range(takes):
            for item in items:
                _, ttfa = synthesize_timed(voice, item["transcript"], seed=take)
                if (row := by_key.get((short_name, item["testcase_id"], take))) is not None:
                    row["ttfa_ms"] = round(ttfa, 1)
        print(f"{short_name}: re-timed", flush=True)
    with results.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        row["errors"], row["ref_words"] = int(row["errors"]), int(row["ref_words"])
        row["ttfa_ms"] = float(row["ttfa_ms"])
    print_table(rows, voices, results)


def print_table(rows, voices, results):
    table = [(name, wer, ttfa, "cloud") for name, wer, ttfa in COVAL_2026_09_08]
    for short_name in voices:
        mine = [r for r in rows if r["voice"] == short_name]
        wer = 100 * sum(r["errors"] for r in mine) / sum(r["ref_words"] for r in mine)
        table.append((f"ToastTTS + {short_name}", round(wer, 1), round(float(np.median([r["ttfa_ms"] for r in mine]))),
                      "this laptop"))
    table.sort(key=lambda t: t[1])
    print(f"\n{'model':<40}{'WER':>7}{'P50 TTFA':>11}  runs on")
    for name, wer, ttfa, where in table:
        ttfa_text = f"{ttfa} ms" if ttfa is not None else "n/a"
        marker = "  <--" if name.startswith("ToastTTS") else ""
        print(f"{name:<40}{wer:>6}%{ttfa_text:>11}  {where}{marker}")
    print("\nCloud numbers: Coval leaderboard 2026-09-08 (14 of 26 models); their TTFA includes the network.")
    print(f"Per-prompt details (what Whisper heard): {results.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
