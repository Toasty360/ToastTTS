"""Pause-distribution eval: the paper's core figure (local, CPU).

For each voice, with the stochastic knobs fixed (@0.3/0.5 so noise_w=0.8
phantom pauses don't contaminate the measurement), synthesize sentences and
measure internal pause durations:

  Test A (targeted): sentences with a SINGLE mid-sentence punctuation mark.
  The longest internal silence is that mark's pause. Reports p50/p95/max
  per punctuation kind, per voice, clamp on/off.

  Test B (general): ~40 mixed sentences; distribution of ALL internal
  silences > 100 ms per voice, clamp on/off.

The clamp is applied as a post-process (toast.pacing.clamp_pauses, 250 ms),
the same guardrail speak_pieces uses when max_pause_ms is set.

  .pause-venv/bin/python distill/eval_pauses.py "voice1" "voice2" [...]

Voice names use the load_voice syntax, e.g.
  piper:en_US-amy_distill_ex02_stage1-best-medium@0.3/0.5
Results -> out/eval/pause_dist_<timestamp>.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toast.pacing import clamp_pauses, find_silences, trim_silence  # noqa: E402
from toast.splitter import split_into_sentences  # noqa: E402
from toast.voices import load_voice  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CLAMP_MS = 250
MIN_SIL_MS = 100  # test B floor: stop closures live below this


COMMA_SENTS = [
    "The cat sat on the mat, and the dog slept quietly.",
    "She opened the window, letting the cool evening air in.",
    "We walked to the store, talking about the game.",
    "He finished his homework, then went outside to play.",
    "The rain stopped at noon, and the sun came out.",
    "They drove through the hills, singing the whole way.",
    "I called her yesterday, but she did not pick up.",
    "The chef chopped the onions, crying a little.",
    "We waited at the station, watching the trains go by.",
    "She saved her money, hoping to buy a bicycle.",
    "The kids played in the yard, laughing loudly.",
    "He read the letter twice, trying to understand it.",
    "The meeting ended early, so we went for coffee.",
    "She wore her new dress, feeling proud.",
    "The dog barked at the mailman, wagging its tail.",
    "We packed our bags, ready for the trip.",
    "He turned off the lights, leaving the room dark.",
    "The teacher smiled, pleased with the class.",
    "They sat by the fire, telling stories.",
    "I lost my keys again, which was frustrating.",
]

SEMICOLON_SENTS = [
    "The storm passed quickly; the streets were dry by noon.",
    "She loves old movies; he prefers anything new.",
    "The cake was perfect; the frosting needed work.",
    "We left at dawn; the roads were empty.",
    "He plays the guitar; she sings along.",
    "The first act dragged; the second half flew by.",
    "I wanted to stay longer; the train would not wait.",
    "The garden is small; every corner is full of flowers.",
    "She trusts him completely; he has never lied to her.",
    "The coffee was strong; it kept me up all night.",
]

COLON_SENTS = [
    "There is one thing to remember: never give up.",
    "He brought three things: a book, a lamp, and a map.",
    "The recipe is simple: mix, bake, and serve.",
    "She had a single goal: to finish the marathon.",
    "The sign said one word: closed.",
    "His answer was short: no.",
    "The plan has two steps: call first, then visit.",
    "There was a knock at the door: the delivery had arrived.",
    "The forecast promised sunshine: a perfect beach day.",
    "He gave me one reason: trust.",
]


def internal_silences(audio, sr, floor_ms=0):
    return [(e - s) / sr * 1000
            for s, e in find_silences(audio, sr)
            if s > 0 and e < len(audio) and (e - s) / sr * 1000 >= floor_ms]


def describe(pauses):
    a = np.array(pauses, dtype=float)
    if len(a) == 0:
        return {"n": 0}
    return {"n": int(len(a)), "p50_ms": round(float(np.median(a)), 1),
            "p95_ms": round(float(np.percentile(a, 95)), 1),
            "max_ms": round(float(a.max()), 1)}


def synth(voice, text, clamp):
    sr = voice.sample_rate
    audio = trim_silence(voice.synthesize(text), sr)
    if clamp:
        audio = clamp_pauses(audio, sr, max_pause_ms=CLAMP_MS)
    return audio, sr


# Piper's duration predictor is stochastic even at controlled noise settings;
# render each sentence REPEATS times and use the median so a single unlucky
# sample doesn't dominate the tail statistics.
REPEATS = 3


def main(voice_names):
    ref = (ROOT / "samples" / "reference.txt").read_text(encoding="utf-8-sig")
    word = (ROOT / "samples" / "word_test.txt").read_text(encoding="utf-8-sig")
    general = (split_into_sentences(ref) + split_into_sentences(word))[:40]
    print(f"{len(general)} general sentences", flush=True)

    results = {"clamp_ms": CLAMP_MS, "noise": "@0.3/0.5 (stochasticity controlled)",
               "repeats": REPEATS, "voices": {}}
    for name in voice_names:
        print(f"\n== {name}", flush=True)
        voice = load_voice(name)
        voice.synthesize("Warm up.")
        vres = {"targeted": {}, "general": {}}
        for clamp in (False, True):
            tag = "clamp_on" if clamp else "clamp_off"
            targeted = {}
            for punct, sents in ((",", COMMA_SENTS), (";", SEMICOLON_SENTS),
                                 (":", COLON_SENTS)):
                pauses = []
                for s in sents:
                    reps = []
                    for _ in range(REPEATS):
                        audio, sr = synth(voice, s, clamp)
                        sil = internal_silences(audio, sr)
                        if sil:
                            reps.append(max(sil))  # the single mark's pause
                    if reps:
                        pauses.append(float(np.median(reps)))
                targeted[punct] = describe(pauses)
            all_sil = []
            for s in general:
                for _ in range(REPEATS):
                    audio, sr = synth(voice, s, clamp)
                    all_sil.extend(internal_silences(audio, sr, floor_ms=MIN_SIL_MS))
            vres["targeted"][tag] = targeted
            vres["general"][tag] = describe(all_sil)
            g = vres["general"][tag]
            print(f"  {tag}: comma {targeted[',']} | general {g}", flush=True)
        results["voices"][name] = vres

    out = ROOT / "out" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"pause_dist_{time.strftime('%Y%m%d-%H%M')}.json"
    path.write_text(json.dumps(results, indent=1))
    print(f"\n-> {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: eval_pauses.py <voice> [<voice> ...]")
    main(sys.argv[1:])
