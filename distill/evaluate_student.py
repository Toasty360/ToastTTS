"""Evaluate distilled students against amy on held-out text (laptop, free).

Success criteria are fixed in research/distillation-plan.md (step 5) BEFORE
looking at results:
  melody     core pitch range >= 9.0 st (plan) and above amy's
  "Well,"    pitch rises before it falls (mid of word above its start)
  endings    last word before a pause no louder than the speech before it (E09 drop <= 0 dB)
  artifacts  UTMOS no more than 0.3 below amy: a BREAKAGE ALARM only, not a naturalness
             judgment (revised before any real student was scored: UTMOS rated ryan 4.41,
             which the listener found unnatural, and called Kokoro and amy equal; D37)
  words      no more wrong words than amy
  one voice  since D39: the student must sound like the target speaker (>= 80% of her own
             consistency) and be no closer to amy than her real recordings are (+0.15);
             author: "I just need a correct voice than 2 voices mixed or overlapped"
  identity   (similarity to amy as such) REPORTED ONLY, not a criterion since D38 (author: "I don't really need
             amy's voice, I just want a proper voice"); similarity to amy and to the
             human speaker are printed so each student's voice can be described
  speed      TTFA and real-time speed within 10% of amy (hard constraint, D32); TTFA also
             allows 5 ms, since ~40 ms timings jitter by a few ms (set on the smoke model,
             before any real student was evaluated)
THE DECIDING TEST IS BLIND LISTENING (distill/blind_test.py): metrics only shortlist.

Held-out text only: DailyTalk conversations with id % 10 == 0 (never trained on)
plus samples/reference.txt and samples/word_test.txt.

  .venv\\Scripts\\python distill\\evaluate_student.py piper:en_US-amy_distill_<run>_<ckpt>-medium [...]
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from e09_phrase_final_softening import final_word_profile  # noqa: E402
from e12_well import describe_word  # noqa: E402
from e13_evaluate import signal_stats, speaker_embedding  # noqa: E402
from e13_vc_feasibility import _dailytalk_clips  # noqa: E402

from toast.metrics import measure  # noqa: E402
from toast.naturalness import naturalness  # noqa: E402
from toast.pacing import speak_naturally, stream_speech, trim_silence  # noqa: E402
from toast.speech_stats import timed_words  # noqa: E402
from toast.splitter import split_into_sentences  # noqa: E402
from toast.text_normalize import normalize_for_speech  # noqa: E402
from toast.voices import load_voice  # noqa: E402
from toast.word_check import transcribe, word_errors  # noqa: E402

N_DAILYTALK = 30
LISTEN = ROOT / "out" / "listen"


def test_set():
    import io

    clips = _dailytalk_clips(N_DAILYTALK)  # held-out conversations only; a "Well," clip first
    human = []
    for _, data in clips:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
        human.append((audio, sr))
    ours = split_into_sentences((ROOT / "samples" / "reference.txt").read_text(encoding="utf-8-sig")) + \
        split_into_sentences((ROOT / "samples" / "word_test.txt").read_text(encoding="utf-8-sig"))
    return [t for t, _ in clips], human, ours


def evaluate(voice, sentences, human_centre, amy_centre):
    sr = voice.sample_rate
    clips = [trim_silence(voice.synthesize(normalize_for_speech(s)), sr) for s in sentences]
    pairs = [(c, sr) for c in clips]
    stats = signal_stats(pairs)
    embs = [speaker_embedding(c, sr) for c in clips]
    stats["sim_amy"] = float(np.mean([e @ amy_centre / np.linalg.norm(amy_centre) for e in embs]))
    stats["sim_human"] = float(np.mean([e @ human_centre / np.linalg.norm(human_centre) for e in embs]))
    stats["wrong"] = sum(word_errors(s, transcribe(c, sr))[0] for s, c in zip(sentences, clips))
    stats["natural"] = float(np.mean([naturalness(c, sr) for c in clips]))
    word, start, end = timed_words(clips[0], sr)[0]
    try:
        stats["well"] = describe_word(clips[0], sr, start, end)
    except Exception as error:
        stats["well"] = f"could not analyse ({type(error).__name__})"
    return stats, clips


def main(student_names):
    dt_sentences, human, ours = test_set()
    reference = (ROOT / "samples" / "reference.txt").read_text(encoding="utf-8-sig").strip()
    human_embs = [speaker_embedding(a, s) for a, s in human]
    human_centre = np.mean(human_embs, axis=0)
    # her own consistency (leave-one-out), the bar for "sounds like one person: her"
    human_self = float(np.mean([e @ (np.sum(human_embs, axis=0) - e) / (len(human_embs) - 1) for e in human_embs]))
    amy = load_voice("amy")
    amy_clips = [trim_silence(amy.synthesize(normalize_for_speech(s)), amy.sample_rate) for s in dt_sentences]
    amy_embs = [speaker_embedding(c, amy.sample_rate) for c in amy_clips]
    amy_centre = np.mean(amy_embs, axis=0)
    amy_self = float(np.mean([e @ (np.sum(amy_embs, axis=0) - e) / (len(amy_embs) - 1) for e in amy_embs]))
    # how much her REAL recordings resemble amy: a single-voice student should be no closer
    human_to_amy = float(np.mean([e @ amy_centre / np.linalg.norm(amy_centre) for e in human_embs]))

    rows = {}
    for name in ["amy", *student_names]:
        voice = load_voice(name)
        voice.synthesize("Warm up.")
        stats, clips = evaluate(voice, dt_sentences, human_centre, amy_centre)
        paragraph = speak_naturally(voice, reference, speed=1.2, seed=0)  # through ToastTTS, as used
        drops, _ = final_word_profile(paragraph, voice.sample_rate)
        stats["drop_db"] = float(np.median(drops))
        _, timing = measure(lambda: stream_speech(voice, reference, speed=1.2), voice.sample_rate, repeats=7)
        stats["ttfa_ms"], stats["x_realtime"] = timing["ttfa_ms"], timing["x_realtime"]
        ours_wrong = sum(word_errors(s, transcribe(trim_silence(voice.synthesize(normalize_for_speech(s)),
                                                                voice.sample_rate), voice.sample_rate))[0]
                         for s in ours)
        stats["wrong"] += ours_wrong
        tag = "amy" if name == "amy" else name.split("amy_distill_")[-1].removesuffix("-medium")
        gap = np.zeros(int(0.4 * voice.sample_rate), dtype=np.float32)
        sf.write(LISTEN / f"80_{tag}_dailytalk.wav", np.concatenate([x for c in clips[:10] for x in (c, gap)]),
                 voice.sample_rate)
        sf.write(LISTEN / f"80_{tag}_paragraph_toast_1.2x.wav", paragraph, voice.sample_rate)
        rows[tag] = stats

    human_stats = signal_stats(human)
    print(f"Held-out test: {len(dt_sentences)} DailyTalk sentences (+{len(ours)} of our own for word errors)")
    print(f"Human DailyTalk speaker: core pitch range {human_stats['core_range']:.2f} st\n")
    print(f"{'voice':<34}{'core st':>8}{'drop dB':>8}{'natural':>8}{'wrong':>6}{'sim amy':>8}{'sim hum':>8}"
          f"{'TTFA ms':>8}{'x rt':>6}")
    for tag, s in rows.items():
        print(f"{tag:<34}{s['core_range']:>8.2f}{s['drop_db']:>8.1f}{s['natural']:>8.2f}{s['wrong']:>6}"
              f"{s['sim_amy']:>8.2f}{s['sim_human']:>8.2f}{s['ttfa_ms']:>8.0f}{s['x_realtime']:>6.1f}")
    print(f"\namy's own consistency: {amy_self:.2f}; the human speaker's own consistency: {human_self:.2f}; "
          f"her real recordings vs amy: {human_to_amy:.2f}")
    print(f"\n\"Well,\" in {dt_sentences[0]!r}:")
    for tag, s in rows.items():
        print(f"  {tag:<34}{s['well']}")

    base = rows["amy"]
    print("\nVerdict (criteria fixed in advance, research/distillation-plan.md step 5):")
    for tag, s in rows.items():
        if tag == "amy":
            continue
        checks = {
            "melody >= 9.0 st": s["core_range"] >= 9.0,
            "melody above amy": s["core_range"] > base["core_range"],
            "soft endings (drop <= 0 dB)": s["drop_db"] <= 0.0,
            "no artifact alarm (UTMOS >= amy - 0.3)": s["natural"] >= base["natural"] - 0.3,
            f"wrong <= amy ({base['wrong']})": s["wrong"] <= base["wrong"],
            # same architecture, so differences are measurement noise; tolerance fixed on the smoke
            # model (44 vs 40 ms) before any real student was scored
            # D39: one clean voice, not two blended (author: "I just need a correct voice than
            # 2 voices mixed or overlapped")
            f"single voice: sounds like her (>= 80% of her own {human_self:.2f})": s["sim_human"] >= 0.8 * human_self,
            f"not blended with amy (<= her real recordings' {human_to_amy:.2f} + 0.15)":
                s["sim_amy"] <= human_to_amy + 0.15,
            "TTFA within 10% (or 5 ms) of amy": s["ttfa_ms"] <= max(1.1 * base["ttfa_ms"], base["ttfa_ms"] + 5),
            "speed within 10% of amy": s["x_realtime"] >= 0.9 * base["x_realtime"],
        }
        print(f"  {tag}: {sum(checks.values())}/{len(checks)} -> " + "; ".join(
            f"{k}: {'yes' if v else 'NO'}" for k, v in checks.items()))
    print("\nMetrics only shortlist. Decide with a blind test: .venv\\Scripts\\python distill\\blind_test.py <student voice>")


if __name__ == "__main__":
    main(sys.argv[1:])
