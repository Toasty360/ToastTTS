"""Benchmark every downloaded English Piper voice (and Kitten, for reference).

For each voice, with our pacing ("smart" split) at speed 0.9:
  - TTFA         time until the first sound can play (median of 3 runs)
  - speed        how many times faster than real time it's made
  - wrong words  Whisper transcribes the reference paragraph (2 takes) and
                 samples/word_test.txt (1 take); every word that comes out
                 wrong, missing or extra counts
  - natural      predicted human rating 1-5 (UTMOS), averaged over the
                 paragraph's sentences

Results go to benchmarks/voice_benchmark.csv, one row per voice, written as
we go. Re-running skips voices already in the file (delete it to redo all).
After changing how words are compared, --rescore recounts from the saved
transcripts without making any audio.
A sample of each voice is saved in out/listen/all_voices/.

  .venv\\Scripts\\python scripts\\bench_voices.py
  .venv\\Scripts\\python scripts\\bench_voices.py --only en_US-lessac-high en_US-ryan-high
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from toast.metrics import measure
from toast.naturalness import naturalness
from toast.pacing import speak_naturally, stream_speech
from toast.splitter import split_into_sentences
from toast.voices import load_voice
from toast.word_check import transcribe, word_errors

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RESULTS = ROOT / "benchmarks" / "voice_benchmark.csv"
SAMPLES = ROOT / "out" / "listen" / "all_voices"
COLUMNS = ["voice", "size_mb", "ttfa_ms", "x_realtime", "wrong_words", "words_checked",
           "natural", "mistakes", "heard"]
SPEED = 0.9


def benchmark(voice_name, reference, word_test):
    voice = load_voice(voice_name)
    voice.synthesize("Warm up.")
    sr = voice.sample_rate

    _, timing = measure(lambda: stream_speech(voice, reference, speed=SPEED), sr, repeats=3)

    heard = []
    for seed, text in enumerate([reference, reference, word_test]):
        audio = speak_naturally(voice, text, speed=SPEED, seed=seed)
        if seed == 0:
            sf.write(SAMPLES / f"{voice_name.replace(':', '_')}.wav", audio, sr)
        heard.append(transcribe(audio, sr))

    scores = [naturalness(speak_naturally(voice, sentence, speed=SPEED), sr)
              for sentence in split_into_sentences(reference)]

    model_file = model_path(voice_name)
    return {
        "voice": voice_name,
        "size_mb": round(model_file.stat().st_size / 1e6) if model_file and model_file.exists() else "",
        "ttfa_ms": round(timing["ttfa_ms"]),
        "x_realtime": round(timing["x_realtime"], 1),
        "natural": round(float(np.mean(scores)), 2),
        "heard": json.dumps(heard),
        **score_words(heard, reference, word_test),
    }


def model_path(voice_name):
    if voice_name.startswith("piper:"):
        return MODELS_DIR / f"{voice_name.removeprefix('piper:')}.onnx"
    if voice_name.startswith("kokoro"):
        precision = voice_name.split(":")[2] if voice_name.count(":") >= 2 else ""
        return MODELS_DIR / "kokoro" / (f"kokoro-v1.0.{precision}.onnx" if precision else "kokoro-v1.0.onnx")
    return None


def score_words(heard, reference, word_test):
    """Compare the three transcripts with the texts they should match."""
    wrong, checked, mistakes = 0, 0, []
    for text, transcript in zip([reference, reference, word_test], heard):
        count, diff = word_errors(text, transcript)
        wrong += count
        checked += len(text.split())
        mistakes += diff
    return {"wrong_words": wrong, "words_checked": checked,
            "mistakes": "; ".join(f"{a}->{b}" for a, b in mistakes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", help="model names to test (default: every en_* model + kitten)")
    parser.add_argument("--rescore", action="store_true",
                        help="recount wrong words from saved transcripts (after changing word_check.py)")
    args = parser.parse_args()

    reference = (ROOT / "samples" / "reference.txt").read_text(encoding="utf-8-sig").strip()
    word_test = (ROOT / "samples" / "word_test.txt").read_text(encoding="utf-8-sig").strip()
    if args.rescore:
        rescore(reference, word_test)
        print_ranking()
        return
    names = args.only or sorted(p.stem for p in MODELS_DIR.glob("en_*.onnx")) + ["kitten"]
    names = [n if n.startswith(("kitten", "piper:", "kokoro")) else f"piper:{n}" for n in names]

    SAMPLES.mkdir(parents=True, exist_ok=True)
    RESULTS.parent.mkdir(exist_ok=True)
    done = set()
    if RESULTS.exists():
        with RESULTS.open(newline="", encoding="utf-8") as f:
            done = {row["voice"] for row in csv.DictReader(f)}

    for n, name in enumerate(names, 1):
        if name in done:
            continue
        started = time.perf_counter()
        try:
            row = benchmark(name, reference, word_test)
        except Exception as error:  # one broken voice shouldn't stop the run
            print(f"[{n}/{len(names)}] {name}: FAILED ({error})")
            continue
        save_row(row)
        print(f"[{n}/{len(names)}] {name:<42} TTFA {row['ttfa_ms']:>4} ms  {row['x_realtime']:>5}x  "
              f"wrong {row['wrong_words']:>2}/{row['words_checked']}  natural {row['natural']}  "
              f"({time.perf_counter() - started:.0f}s)", flush=True)

    print_ranking()


def save_row(row):
    # Excel locks a CSV while it's open; wait for it instead of losing the result.
    warned = False
    while True:
        try:
            new_file = not RESULTS.exists()
            with RESULTS.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
            return
        except PermissionError:
            if not warned:
                print(f"  {RESULTS.name} is open in another program (Excel?); close it and I'll continue.", flush=True)
                warned = True
            time.sleep(5)


def rescore(reference, word_test):
    with RESULTS.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row.update(score_words(json.loads(row["heard"]), reference, word_test))
    with RESULTS.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_ranking():
    with RESULTS.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (int(r["wrong_words"]) > 2, -float(r["natural"])))
    print(f"\n{'voice':<42}{'natural':>8}{'wrong':>7}{'TTFA':>9}{'speed':>8}{'MB':>6}")
    for r in rows:
        print(f"{r['voice']:<42}{r['natural']:>8}{r['wrong_words']:>7}{r['ttfa_ms']:>7}ms"
              f"{r['x_realtime']:>7}x{r['size_mb']:>6}")
    print("\nSorted by naturalness, voices with more than 2 wrong words last.")


if __name__ == "__main__":
    main()
