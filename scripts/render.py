"""Make recordings of the reference paragraph and measure each one.

Every recording gets timed (time to first audio, stalls, speed), printed
in a table and added to benchmarks/results.csv so versions can be compared
over time.

With no options it makes the current listening set in out/listen/:

  .venv\\Scripts\\python scripts\\render.py

Or one recording with your own settings:

  .venv\\Scripts\\python scripts\\render.py --voice amy --split smart --speed 1.2 --play

Voices: "amy" (default, speed 1.2), "lessac", "piper:en_US-lessac-high", "3922",
"3922@0.5/0.8" (with randomness settings), "kitten", "kitten:micro". Splits: smart, clause, sentence, whole.
"""

import argparse
from pathlib import Path

import soundfile as sf

from toast.metrics import log_result, measure
from toast.pacing import stream_speech
from toast.voices import load_voice

ROOT = Path(__file__).resolve().parents[1]
LISTEN_DIR = ROOT / "out" / "listen"

# (file name, voice, split, speed). "whole" = the model says everything in
# one go, the way plain TTS works.
LISTENING_SET = [
    ("6_kitten_paused_sentences", "kitten", "sentence", 1.0),  # the one to beat
    ("12_lessac_medium_smart", "lessac", "smart", 0.9),
    ("16_amy_1.2x", "amy", "smart", 1.2),                       # our default: cloud-voice speaking rate
]


def speech_chunks(voice, text, split, speed):
    if split == "whole":
        yield voice.synthesize(text, speed=speed)
    else:
        yield from stream_speech(voice, text, split=split, speed=speed)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", type=Path, default=ROOT / "samples" / "reference.txt")
    parser.add_argument("--voice", help="make one recording with this voice instead of the listening set")
    parser.add_argument("--split", choices=["smart", "clause", "sentence", "whole"], default="smart")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--repeat", type=int, default=3, help="runs per recording for steadier timings")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--check-words", action="store_true",
                        help="transcribe each recording with Whisper and count wrong words (slower)")
    args = parser.parse_args()
    text = args.text.read_text(encoding="utf-8-sig").strip()

    if args.voice:
        name = f"custom_{args.voice.replace(':', '_').replace('/', '-')}_{args.split}_{args.speed}"
        jobs = [(name, args.voice, args.split, args.speed)]
    else:
        jobs = LISTENING_SET
    LISTEN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'recording':<34}{'TTFA':>9}{'stalls':>8}{'made in':>9}{'length':>8}{'speed':>8}"
          + (f"{'wrong words':>13}" if args.check_words else ""))
    voices = {}
    for file_name, voice_name, split, speed in jobs:
        if voice_name not in voices:
            voices[voice_name] = load_voice(voice_name)
            voices[voice_name].synthesize("Warm up.")  # the first call is always slower
        voice = voices[voice_name]

        repeats = 1 if voice_name.startswith("kitten") else args.repeat  # Kitten is slow
        audio, stats = measure(lambda: speech_chunks(voice, text, split, speed), voice.sample_rate, repeats)
        sf.write(LISTEN_DIR / f"{file_name}.wav", audio, voice.sample_rate)

        mistakes = []
        if args.check_words:
            from toast.word_check import transcribe, word_errors

            stats["word_errors"], mistakes = word_errors(text, transcribe(audio, voice.sample_rate))
        log_result(file_name, voice.name, split, speed, stats)

        print(f"{file_name:<34}{stats['ttfa_ms']:>7.0f}ms{stats['stalls']:>8}"
              f"{stats['made_in_s']:>8.1f}s{stats['audio_s']:>7.1f}s{stats['x_realtime']:>7.1f}x"
              + (f"{stats['word_errors']:>13}" if args.check_words else ""))
        if mistakes:
            print("      heard: " + ", ".join(f"{a}->{b}" for a, b in mistakes))

        if args.play:
            from toast.player import play

            play(audio, voice.sample_rate)

    print()
    print("  TTFA   = time until the first sound can play (lower is better)")
    print("  stalls = times playback would run out and wait for the next piece (want 0)")
    print("  speed  = how many times faster than real time it was made")
    if args.check_words:
        print("  wrong words = words Whisper heard differently (it also writes '3:15 PM'")
        print("                oddly sometimes, so treat 1-2 as noise)")
    print("  All runs are saved in benchmarks/results.csv")


if __name__ == "__main__":
    main()
