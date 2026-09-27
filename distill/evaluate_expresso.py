"""Evaluate an Expresso-fine-tuned student against amy (laptop, free).

Adapted from distill/evaluate_student.py for the Expresso non-commercial run:
the human reference is the chosen Expresso speaker's own gated training clips,
not DailyTalk. 40 test sentences: 30 held-out DailyTalk + 5 reference + 5 word_test.

Success criteria (from the run brief, fixed before scoring):
  words      no more wrong words than amy (notebook 15 baseline: 7/40)
  artifacts  UTMOS no more than 0.3 below amy (baseline 4.42): breakage alarm only
  one voice  student sounds like the Expresso speaker (>= 80% of her own
             consistency) and is no closer to amy than her real recordings are (+0.15)
  speed      TTFA and real-time speed within 10% of amy (hard constraint)

Listening decides; metrics only shortlist.

  .venv/Scripts/python distill/evaluate_expresso.py piper:en_US-amy_expresso_<spk>-medium <spk>
"""

import csv
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from e13_evaluate import signal_stats, speaker_embedding  # noqa: E402
try:
    from e13_vc_feasibility import _dailytalk_clips  # noqa: E402
    # Probe the data file now: if DailyTalk isn't on this machine, fall through
    # to the Expresso-text fallback below instead of failing mid-eval.
    from e13_vc_feasibility import DAILYTALK  # noqa: E402
    import os as _os
    if not _os.path.exists(DAILYTALK):
        raise ImportError("dailytalk parquet not on this machine")
except ImportError:  # e13 imports modal at module level; use held-out Expresso texts
    def _dailytalk_clips(count):
        import csv
        gate = ROOT / "distill" / "data" / "train" / "expresso_ex02" / "gate.csv"
        dropped = [r for r in csv.DictReader(gate.open(), delimiter="|") if r["kept"] != "True"]
        random.seed(11)
        return [(r["text"], None) for r in random.sample(dropped, min(count, len(dropped)))]

from toast.metrics import measure  # noqa: E402
from toast.naturalness import naturalness  # noqa: E402
from toast.pacing import stream_speech, trim_silence  # noqa: E402
from toast.splitter import split_into_sentences  # noqa: E402
from toast.text_normalize import normalize_for_speech  # noqa: E402
from toast.voices import load_voice  # noqa: E402
from toast.word_check import transcribe, word_errors  # noqa: E402

N_DAILYTALK = 30
N_HUMAN = 30
LISTEN = ROOT / "out" / "listen"


def test_sentences():
    import io

    clips = _dailytalk_clips(N_DAILYTALK)
    ours = split_into_sentences((ROOT / "samples" / "reference.txt").read_text(encoding="utf-8-sig")) + \
        split_into_sentences((ROOT / "samples" / "word_test.txt").read_text(encoding="utf-8-sig"))
    return [t for t, _ in clips] + ours


def human_clips(speaker):
    """N_HUMAN random kept clips of the Expresso speaker, as (audio, sr)."""
    train = ROOT / "distill" / "data" / "train" / f"expresso_{speaker}"
    kept = []
    with open(train / "gate.csv", newline="") as f:
        for row in csv.DictReader(f, delimiter="|"):
            if row["kept"] == "True":
                kept.append(row["id"])
    random.seed(7)
    clips = []
    for clip_id in random.sample(kept, min(N_HUMAN, len(kept))):
        audio, sr = sf.read(train / "wavs" / f"{clip_id}.wav", dtype="float32")
        clips.append((audio, sr))
    return clips


def evaluate(voice, sentences, human_centre, amy_centre):
    sr = voice.sample_rate
    clips = [trim_silence(voice.synthesize(normalize_for_speech(s)), sr) for s in sentences]
    embs = [speaker_embedding(c, sr) for c in clips]
    stats = {
        "sim_amy": float(np.mean([e @ amy_centre / np.linalg.norm(amy_centre) for e in embs])),
        "sim_human": float(np.mean([e @ human_centre / np.linalg.norm(human_centre) for e in embs])),
        "wrong": sum(word_errors(s, transcribe(c, sr))[0] for s, c in zip(sentences, clips)),
        "natural": float(np.mean([naturalness(c, sr) for c in clips])),
    }
    return stats, clips


def main(student_name, speaker):
    sentences = test_sentences()
    human = human_clips(speaker)
    human_embs = [speaker_embedding(a, s) for a, s in human]
    human_centre = np.mean(human_embs, axis=0)
    human_self = float(np.mean([e @ (np.sum(human_embs, axis=0) - e) / (len(human_embs) - 1)
                                for e in human_embs]))
    amy = load_voice("amy")
    amy_clips = [trim_silence(amy.synthesize(normalize_for_speech(s)), amy.sample_rate)
                 for s in sentences]
    amy_embs = [speaker_embedding(c, amy.sample_rate) for c in amy_clips]
    amy_centre = np.mean(amy_embs, axis=0)
    amy_self = float(np.mean([e @ (np.sum(amy_embs, axis=0) - e) / (len(amy_embs) - 1)
                              for e in amy_embs]))
    human_to_amy = float(np.mean([e @ amy_centre / np.linalg.norm(amy_centre) for e in human_embs]))

    rows = {}
    for name in ["amy", student_name]:
        voice = load_voice(name)
        voice.synthesize("Warm up.")
        stats, clips = evaluate(voice, sentences, human_centre, amy_centre)
        _, timing = measure(lambda: stream_speech(voice, sentences[0]), voice.sample_rate, repeats=7)
        stats["ttfa_ms"], stats["x_realtime"] = timing["ttfa_ms"], timing["x_realtime"]
        tag = "amy" if name == "amy" else student_name.split("amy_")[-1].removesuffix("-medium")
        tag = "".join(c if c.isalnum() or c in "-_." else "_" for c in tag)  # safe filename
        gap = np.zeros(int(0.4 * voice.sample_rate), dtype=np.float32)
        LISTEN.mkdir(parents=True, exist_ok=True)
        sf.write(LISTEN / f"81_{tag}_40sent.wav",
                 np.concatenate([x for c in clips for x in (c, gap)]), voice.sample_rate)
        rows[tag] = stats

    print(f"Test: {len(sentences)} sentences; human reference: {len(human)} {speaker} clips")
    print(f"{'voice':<34}{'wrong':>6}{'natural':>8}{'sim amy':>8}{'sim hum':>8}"
          f"{'TTFA ms':>8}{'x rt':>6}")
    for tag, s in rows.items():
        print(f"{tag:<34}{s['wrong']:>6}{s['natural']:>8.2f}{s['sim_amy']:>8.2f}"
              f"{s['sim_human']:>8.2f}{s['ttfa_ms']:>8.0f}{s['x_realtime']:>6.1f}")
    print(f"\namy's own consistency: {amy_self:.2f}; {speaker}'s own consistency: {human_self:.2f}; "
          f"her real recordings vs amy: {human_to_amy:.2f}")

    base = rows["amy"]
    print("\nVerdict:")
    for tag, s in rows.items():
        if tag == "amy":
            continue
        checks = {
            f"wrong <= amy ({base['wrong']})": s["wrong"] <= base["wrong"],
            "no artifact alarm (UTMOS >= amy - 0.3)": s["natural"] >= base["natural"] - 0.3,
            f"single voice: sounds like her (>= 80% of her own {human_self:.2f})":
                s["sim_human"] >= 0.8 * human_self,
            f"not blended with amy (<= her real {human_to_amy:.2f} + 0.15)":
                s["sim_amy"] <= human_to_amy + 0.15,
            "TTFA within 10% (or 5 ms) of amy":
                s["ttfa_ms"] <= max(1.1 * base["ttfa_ms"], base["ttfa_ms"] + 5),
            "speed within 10% of amy": s["x_realtime"] >= 0.9 * base["x_realtime"],
        }
        print(f"  {tag}: {sum(checks.values())}/{len(checks)} -> " + "; ".join(
            f"{k}: {'yes' if v else 'NO'}" for k, v in checks.items()))
    print("\nMetrics only shortlist. Decide with a blind test: "
          ".venv/Scripts/python distill/blind_test.py <student voice>")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
