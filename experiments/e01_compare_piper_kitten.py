"""E01: Piper vs KittenTTS, whole paragraph vs split into pieces.

The first experiment of the project (see research/notebook/01-baseline.md).
For each model this makes two recordings in out/e01/:
  <model>_whole.wav   the paragraph spoken in one go (how TTS normally works)
  <model>_pieces.wav  each piece spoken separately, glued back to back,
                      with no pauses or fades

Question: does a model still sound natural when fed short pieces? (Finding:
Kitten's "whole" sounded good, its "pieces" choppy: each fragment gets
sentence-final intonation and padding.)

Usage:
  .venv\\Scripts\\python experiments\\e01_compare_piper_kitten.py            # piper and kitten
  .venv\\Scripts\\python experiments\\e01_compare_piper_kitten.py --play     # also play them
  .venv\\Scripts\\python experiments\\e01_compare_piper_kitten.py piper kitten:micro kitten:mini:Luna
"""

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from toast.splitter import split_into_pieces
from toast.voices import load_voice

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "out" / "e01"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("voices", nargs="*", default=["piper", "kitten"])
    parser.add_argument("--text", type=Path, default=ROOT / "samples" / "reference.txt")
    parser.add_argument("--play", action="store_true", help="play each recording after making it")
    args = parser.parse_args()

    text = args.text.read_text(encoding="utf-8-sig").strip()
    pieces = split_into_pieces(text)
    OUT_DIR.mkdir(exist_ok=True)

    print(f"{len(pieces)} pieces:")
    for i, piece in enumerate(pieces, 1):
        print(f"  {i:2}. {piece}")
    print()

    results = []
    for voice_name in args.voices:
        print(f"Loading {voice_name} ...")
        start = time.perf_counter()
        voice = load_voice(voice_name)
        load_seconds = time.perf_counter() - start
        voice.synthesize("Warm up.")  # the first call is always slower

        start = time.perf_counter()
        whole = voice.synthesize(text)
        whole_seconds = time.perf_counter() - start

        piece_audio = []
        start = time.perf_counter()
        for piece in pieces:
            piece_audio.append(voice.synthesize(piece))
            if len(piece_audio) == 1:
                first_piece_seconds = time.perf_counter() - start
        pieces_seconds = time.perf_counter() - start

        safe_name = voice.name.replace(":", "_")
        whole_path = OUT_DIR / f"{safe_name}_whole.wav"
        pieces_path = OUT_DIR / f"{safe_name}_pieces.wav"
        sf.write(whole_path, whole, voice.sample_rate)
        sf.write(pieces_path, np.concatenate(piece_audio), voice.sample_rate)

        audio_seconds = len(whole) / voice.sample_rate
        results.append((voice.name, load_seconds, whole_seconds, first_piece_seconds,
                        whole_seconds / audio_seconds, pieces_seconds))

        if args.play:
            from toast.player import play

            for label, audio in (("whole", whole), ("pieces", np.concatenate(piece_audio))):
                print(f"  playing {voice.name} ({label}) ...")
                play(audio, voice.sample_rate)

    print()
    print(f"{'model':<34}{'load':>8}{'whole':>9}{'1st piece':>11}{'speed*':>9}")
    for name, load_s, whole_s, first_s, rtf, _ in results:
        print(f"{name:<34}{load_s:>7.1f}s{whole_s:>8.2f}s{first_s * 1000:>9.0f}ms{1 / rtf:>8.1f}x")
    print()
    print("  whole     = time to make the full paragraph in one go")
    print("  1st piece = time before the first bit of audio could start playing")
    print("  speed*    = how many times faster than real time the model runs")
    print(f"\nRecordings saved in {OUT_DIR}")


if __name__ == "__main__":
    main()
