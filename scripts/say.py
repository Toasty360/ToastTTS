"""Speak any text with ToastTTS: typed, passed as an argument, or piped in.

  .venv\\Scripts\\python scripts\\say.py                       # interactive: type a line, hear it
  .venv\\Scripts\\python scripts\\say.py "Well, hi there! How can I help?"
  .venv\\Scripts\\python scripts\\say.py "Order #4521 ships at 3:15 PM." --save out\\say.wav --no-play
  some_llm_cli "tell me a joke" | .venv\\Scripts\\python scripts\\say.py -   # speaks while text streams in

Every run prints the time to first audio and how the text was cut into pieces.
"""

import argparse
import sys
from pathlib import Path

from toast.engine import DEFAULT_SPEED, DEFAULT_VOICE, ToastEngine


def report(result):
    print(f"  TTFA {result.ttfa_ms:.0f} ms | {result.seconds:.1f} s of audio made in {result.made_in_s:.2f} s")
    print("  pieces: " + " | ".join(result.pieces))


def stdin_stream():
    # Hand text over as it arrives, a character at a time, like LLM tokens.
    # PowerShell prefixes piped text with an invisible byte-order mark; drop it.
    return (char for char in iter(lambda: sys.stdin.read(1), "") if char != "﻿")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="*", help="text to speak; '-' reads a stream from stdin; none = interactive")
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--save", type=Path, help="also write the audio to this .wav file")
    parser.add_argument("--no-play", action="store_true", help="don't play, just make (and --save) the audio")
    parser.add_argument("--wake", type=float, help="seconds the speakers run before the first word (default 0.6)")
    parser.add_argument("--idle-db", type=float, help="keep-awake hiss level when idle (default -60)")
    parser.add_argument("--attach-openers", action="store_true",
                        help="keep \"Well,\" / \"So,\" with the next words: more natural, slower start")
    args = parser.parse_args()

    tts = ToastEngine(args.voice, args.speed, open_audio=not args.no_play, wake_s=args.wake, idle_db=args.idle_db,
                      attach_openers=args.attach_openers)

    def run(text):
        result = tts.run(text, play=not args.no_play)
        if args.save:
            import soundfile as sf

            args.save.parent.mkdir(parents=True, exist_ok=True)
            sf.write(args.save, result.audio, tts.sample_rate)
            print(f"  saved {args.save}")
        report(result)

    with tts:
        if args.text == ["-"] or (not args.text and not sys.stdin.isatty()):
            run(stdin_stream())
        elif args.text:
            run(" ".join(args.text))
        else:
            print(f"ToastTTS ({tts.voice.name}, speed {args.speed}). Type a line and press Enter; empty line to quit.")
            while (line := input("> ").strip()):
                run(line)


if __name__ == "__main__":
    main()
