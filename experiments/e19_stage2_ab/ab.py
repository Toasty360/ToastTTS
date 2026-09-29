#!/usr/bin/env python3
"""e19: blind A/B pick between the two stage-2 finalist checkpoints.

Stage 2 of the Expresso ex02 fine-tune produced two near-identical
finalists whose metrics cannot separate them:

  leg1  epoch 7059, val_mos 4.2126  (leg-1 best)
  leg2  epoch 7214, val_mos 4.1975  (leg-2 best)

The standard 40-sentence eval was run on the leg-2 voice (11 word errors
vs amy's 12, UTMOS 4.25 vs 4.35); the leg-1 best was not separately
evaluated because the validation scores differ by 0.015. The pick is made
by ear, blind.

Fairness rules (the point of this experiment): the ONLY thing that differs
between the two files of a pair is the checkpoint. For each sentence:

  - same sentence, same voice settings (noise_scale=0.3, noise_w=0.5),
    same speed (1.0), same ToastTTS split ("smart")
  - same seed, so the piece plan and the hand-written pause-table draws
    are identical for both renders
  - no pause-model override, no pause clamp

The A/B assignment is shuffled per sentence (rng seed SHUFFLE_SEED). The
key is written to key.json, which must NOT be opened until the answers
are in.

Usage:
    ~/workspace/.pause-venv/bin/python ab.py
    # writes wav/01_A.wav ... wav/10_B.wav, key.json, sentences.txt, answers.txt

    ~/workspace/.pause-venv/bin/python ab.py --score
    # reads answers.txt, reveals the key, tallies with a sign test
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from toast.pacing import speak_naturally  # noqa: E402
from toast.splitter import split_into_sentences  # noqa: E402
from toast.voices import load_voice  # noqa: E402

VOICES = {
    # epoch 7059, val_mos 4.2126 (leg-1 best)
    "leg1": "piper:en_US-amy_distill_ex02_stage2-leg1-best-medium@0.3/0.5",
    # epoch 7214, val_mos 4.1975 (leg-2 best)
    "leg2": "piper:en_US-amy_distill_ex02_stage2-best-medium@0.3/0.5",
}
SPEED = 1.0
SHUFFLE_SEED = 7

OUTDIR = Path(__file__).resolve().parent / "wav"


def test_sentences():
    ref = (REPO / "samples" / "reference.txt").read_text(encoding="utf-8-sig")
    wt = (REPO / "samples" / "word_test.txt").read_text(encoding="utf-8-sig")
    return split_into_sentences(ref) + split_into_sentences(wt)  # 5 + 5


def make():
    OUTDIR.mkdir(exist_ok=True)
    sentences = test_sentences()
    voices = {name: load_voice(spec) for name, spec in VOICES.items()}
    rng = random.Random(SHUFFLE_SEED)
    key = {
        "voices": VOICES,
        "speed": SPEED,
        "noise": "0.3/0.5",
        "rendered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "items": [],
    }
    for n, sentence in enumerate(sentences, 1):
        order = list(VOICES)
        rng.shuffle(order)
        for letter, who in zip("AB", order):
            voice = voices[who]
            audio = speak_naturally(voice, sentence, speed=SPEED, seed=n)
            sf.write(OUTDIR / f"{n:02d}_{letter}.wav", audio, voice.sample_rate)
        key["items"].append({"n": n, "A": order[0], "B": order[1], "sentence": sentence})
        print(f"  {n:02d}: A={order[0]} B={order[1]}  {sentence[:60]}...")
    (OUTDIR / "key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    (OUTDIR / "sentences.txt").write_text(
        "\n".join(f"{i['n']:02d}. {i['sentence']}" for i in key["items"]), encoding="utf-8")
    (OUTDIR / "answers.txt").write_text("", encoding="utf-8")
    print(f"\n{len(sentences)} pairs in {OUTDIR}\n"
          "Listen to each <n>_A.wav / <n>_B.wav; in answers.txt write one line per pair: A, B or same\n"
          "(which sounds more natural?). Don't open key.json. Then run with --score.")


def score():
    key = json.loads((OUTDIR / "key.json").read_text(encoding="utf-8"))
    answers = [line.strip().upper() for line in (OUTDIR / "answers.txt").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    tally = {"leg1": 0, "leg2": 0, "same": 0}
    for item, answer in zip(key["items"], answers):
        tally["same" if answer == "SAME" else item[answer]] += 1
    decided = tally["leg1"] + tally["leg2"]
    print(f"{len(answers)} answers: leg1 preferred {tally['leg1']}, "
          f"leg2 preferred {tally['leg2']}, same {tally['same']}")
    if decided:
        # two-sided sign test: how likely is a split at least this lopsided by chance?
        from math import comb
        k = max(tally["leg1"], tally["leg2"])
        p = min(1.0, 2 * sum(comb(decided, i) for i in range(k, decided + 1)) / 2 ** decided)
        print(f"sign test p = {p:.3f} ({'a real preference' if p < 0.05 else 'could be chance'} at 0.05)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score", action="store_true", help="tally answers.txt against key.json")
    args = parser.parse_args()
    if args.score:
        score()
    else:
        make()


if __name__ == "__main__":
    main()
