"""Blind A/B listening test: the deciding test for a distilled student.

UTMOS can't be trusted to judge naturalness for this listener (it rated ryan
4.41, which sounded unnatural, and scored Kokoro and amy the same), so the
decision is made by ear, without knowing which file is which.

For each of N held-out sentences, amy and the student both speak it through
ToastTTS (same pauses: same seed). They're saved as <n>_A.wav / <n>_B.wav with
the order randomised per sentence. The key is written to key.json, which
should NOT be opened until the answers are in.

The listener writes answers.txt: one line per sentence, "A", "B" or "same"
for "which sounds more natural?". Then `--score` reveals the key and counts.

  .venv\\Scripts\\python distill\\blind_test.py piper:en_US-amy_distill_<run>_<ckpt>-medium
  .venv\\Scripts\\python distill\\blind_test.py --score out\\blind\\<test>
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from e13_vc_feasibility import _dailytalk_clips  # noqa: E402

from toast.pacing import speak_naturally  # noqa: E402
from toast.voices import load_voice  # noqa: E402


def make(student, count, speed, seed):
    folder = ROOT / "out" / "blind" / time.strftime("%Y%m%d-%H%M")
    folder.mkdir(parents=True, exist_ok=True)
    # held-out DailyTalk lines, skipping the first 10 (used in metric reports and listening files)
    sentences = [text for text, _ in _dailytalk_clips(count + 10)][10:]
    voices = {"amy": load_voice("amy"), "student": load_voice(student)}
    rng = random.Random(seed)
    key = {"student": student, "speed": speed, "items": []}
    for n, sentence in enumerate(sentences, 1):
        order = ["amy", "student"]
        rng.shuffle(order)
        for letter, who in zip("AB", order):
            voice = voices[who]
            sf.write(folder / f"{n:02d}_{letter}.wav", speak_naturally(voice, sentence, speed=speed, seed=n),
                     voice.sample_rate)
        key["items"].append({"n": n, "A": order[0], "B": order[1], "sentence": sentence})
    (folder / "key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    (folder / "sentences.txt").write_text("\n".join(f"{i['n']:02d}. {i['sentence']}" for i in key["items"]),
                                          encoding="utf-8")
    (folder / "answers.txt").write_text("", encoding="utf-8")
    print(f"{len(sentences)} pairs in {folder}\n"
          "Listen to each <n>_A.wav / <n>_B.wav; in answers.txt write one line per pair: A, B or same\n"
          "(which sounds more natural?). Don't open key.json. Then run with --score.")
    return folder


def score(folder):
    folder = Path(folder)
    key = json.loads((folder / "key.json").read_text(encoding="utf-8"))
    answers = [line.strip().upper() for line in (folder / "answers.txt").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    tally = {"student": 0, "amy": 0, "same": 0}
    for item, answer in zip(key["items"], answers):
        tally["same" if answer == "SAME" else item[answer]] += 1
    decided = tally["student"] + tally["amy"]
    print(f"{len(answers)} answers for {key['student']}: student preferred {tally['student']}, "
          f"amy preferred {tally['amy']}, same {tally['same']}")
    if decided:
        # two-sided sign test: how likely is a split at least this lopsided by chance?
        from math import comb
        k = max(tally["student"], tally["amy"])
        p = min(1.0, 2 * sum(comb(decided, i) for i in range(k, decided + 1)) / 2 ** decided)
        print(f"sign test p = {p:.3f} ({'a real preference' if p < 0.05 else 'could be chance'} at 0.05)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("student", nargs="?")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--speed", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--score", help="folder of a finished test")
    args = parser.parse_args()
    if args.score:
        score(args.score)
    else:
        make(args.student, args.count, args.speed, args.seed)


if __name__ == "__main__":
    main()
