"""Speak an LLM reply while it is still being written.

A fake LLM streams the reference paragraph in word fragments at a steady
rate (30 tokens/second, typical for a fast chat model). The live splitter
cuts pieces as soon as it's sure about a pause point, Piper (amy voice) speaks them,
and the audio is played (with --play) as it's made.

TTFA here is measured from the moment the FIRST TOKEN arrives, so it
includes waiting for enough words to make the first piece.

  .venv\\Scripts\\python scripts\\live.py --play
  .venv\\Scripts\\python scripts\\live.py --tokens-per-second 10 --flash-words 3
"""

import argparse
import time
from pathlib import Path

import soundfile as sf

from toast.fake_llm import fake_llm_tokens
from toast.metrics import log_result, measure
from toast.pacing import stream_from_llm
from toast.player import LivePlayer, play_while_making
from toast.stream_chunker import StreamChunker
from toast.voices import load_voice

ROOT = Path(__file__).resolve().parents[1]


class TimedChunker(StreamChunker):
    """A StreamChunker that notes when each piece was cut."""

    def __init__(self, **options):
        super().__init__(**options)
        self.cut_times = []

    def _take(self, cut):
        piece = super()._take(cut)
        self.cut_times.append((time.perf_counter(), piece[0]))
        return piece


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", type=Path, default=ROOT / "samples" / "reference.txt")
    parser.add_argument("--voice", default="amy")
    parser.add_argument("--speed", type=float, default=1.2)
    parser.add_argument("--tokens-per-second", type=float, default=30.0)
    parser.add_argument("--flash-words", type=int, default=4,
                        help="max words to wait for before the first piece, if no comma comes sooner")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--play", action="store_true", help="play it live (runs once)")
    args = parser.parse_args()

    text = args.text.read_text(encoding="utf-8-sig").strip()
    voice = load_voice(args.voice)
    voice.synthesize("Warm up.")  # the first call is always slower
    repeats = 1 if args.play else args.repeat
    chunkers = []
    player = None
    if args.play:
        player = LivePlayer(voice.sample_rate)  # opened before timing starts,
        player.wait_until_awake()               # and awake before the first word

    def make_chunks():
        chunker = TimedChunker(flash_words=args.flash_words)
        chunkers.append(chunker)
        tokens = fake_llm_tokens(text, args.tokens_per_second)
        chunks = stream_from_llm(voice, tokens, speed=args.speed, chunker=chunker)
        chunker.started = time.perf_counter()
        return play_while_making(chunks, voice.sample_rate, player) if args.play else chunks

    audio, stats = measure(make_chunks, voice.sample_rate, repeats)
    if player:
        player.close()

    first = chunkers[0]
    print(f"Pieces as they were cut (LLM at {args.tokens_per_second:g} tokens/s):")
    for moment, piece in first.cut_times:
        print(f"  {(moment - first.started) * 1000:6.0f} ms  {piece}")

    label = f"live_{args.tokens_per_second:g}tps_flash{args.flash_words}"
    out_path = ROOT / "out" / "listen" / f"10_{label}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio, voice.sample_rate)
    log_result(label, voice.name, f"live@{args.tokens_per_second:g}tps", args.speed, stats)

    print()
    print(f"TTFA (from first token): {stats['ttfa_ms']:.0f} ms")
    print(f"Stalls:                  {stats['stalls']} ({stats['stall_ms']:.0f} ms total)")
    print(f"Saved {out_path.relative_to(ROOT)}; logged to benchmarks/results.csv")


if __name__ == "__main__":
    main()
