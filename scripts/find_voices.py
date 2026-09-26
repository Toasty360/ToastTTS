"""Find the clearest speakers in a multi-speaker Piper model (904 speakers).

Each speaker reads a few tricky sentences; Whisper writes down what it
heard; we count wrong words. The clearest speakers get a sample saved to
out/listen/voices/ so you only have to listen to a shortlist.

  .venv\\Scripts\\python scripts\\find_voices.py --count 60
  .venv\\Scripts\\python scripts\\find_voices.py --count 904     # every speaker (slow)
  .venv\\Scripts\\python scripts\\find_voices.py --model en_US-libritts-high
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from toast.pacing import speak_naturally
from toast.voices import PiperEngine
from toast.word_check import transcribe, word_errors

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

TEST_SENTENCES = [
    "Does it handle subtle micro-breaks, like after a semicolon?",
    "It's about cadence, rhythm, and natural breathing.",
    "The listener will notice immediately, rushing through every single clause.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="en_US-libritts_r-medium")
    parser.add_argument("--count", type=int, default=60, help="how many speakers to test, spread across all 904")
    parser.add_argument("--noise", default="0.5/0.8", help="randomness settings, as in '3922@0.5/0.8'")
    parser.add_argument("--speed", type=float, default=0.9)
    parser.add_argument("--shortlist", type=int, default=8)
    args = parser.parse_args()
    noise_scale, noise_w = (float(x) for x in args.noise.split("/"))

    speaker_map = json.loads((MODELS_DIR / f"{args.model}.onnx.json").read_text(encoding="utf-8"))["speaker_id_map"]
    names = sorted(speaker_map, key=speaker_map.get)
    picks = [names[int(i)] for i in np.linspace(0, len(names) - 1, min(args.count, len(names)))]
    usual = "3922" if "3922" in speaker_map else "p3922"
    if usual not in picks:
        picks.insert(0, usual)  # always include the 3922 speaker for comparison
    total_words = sum(len(s.split()) for s in TEST_SENTENCES)

    results = []
    started = time.perf_counter()
    for n, name in enumerate(picks, 1):
        voice = PiperEngine(args.model, name, noise_scale=noise_scale, noise_w_scale=noise_w)
        errors, mistakes = 0, []
        for sentence in TEST_SENTENCES:
            heard = transcribe(voice.synthesize(sentence, speed=args.speed), voice.sample_rate)
            count, diff = word_errors(sentence, heard)
            errors += count
            mistakes += diff
        results.append((errors / total_words, name, mistakes))
        shown = ", ".join(f"{a}->{b}" for a, b in mistakes[:3])
        print(f"[{n:3}/{len(picks)}] speaker {name:>5}: {errors} wrong words  {shown}")

    results.sort(key=lambda r: r[0])
    out_csv = ROOT / "benchmarks" / f"voice_scan_{args.model}.csv"
    out_csv.parent.mkdir(exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["speaker", "word_error_rate", "mistakes"])
        for rate, name, mistakes in results:
            writer.writerow([name, round(rate, 3), "; ".join(f"{a}->{b}" for a, b in mistakes)])

    sample_dir = ROOT / "out" / "listen" / "voices" / args.model
    sample_dir.mkdir(parents=True, exist_ok=True)
    text = (ROOT / "samples" / "reference.txt").read_text(encoding="utf-8-sig").strip()
    print(f"\nClearest {args.shortlist} (samples in out/listen/voices/):")
    for rank, (rate, name, _) in enumerate(results[:args.shortlist], 1):
        voice = PiperEngine(args.model, name, noise_scale=noise_scale, noise_w_scale=noise_w)
        audio = speak_naturally(voice, text, speed=args.speed)
        sf.write(sample_dir / f"{rank}_speaker_{name}.wav", audio, voice.sample_rate)
        print(f"  {rank}. speaker {name:>5}  word error rate {rate:.0%}")
    rate_3922 = next(r for r, name, _ in results if name == usual)
    print(f"\n  (3922 for comparison: word error rate {rate_3922:.0%})")
    print(f"Took {time.perf_counter() - started:.0f}s. Full list: {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
