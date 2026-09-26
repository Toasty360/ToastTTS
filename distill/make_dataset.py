"""Quality-gate the converted clips and build a Piper training set (laptop, free).

A clip only teaches amy if the conversion is clean. For every converted clip:
  words      Whisper (small.en) must hear the text. A mismatch is forgiven only if
             the ORIGINAL human clip has the same number of mismatches (then it's
             Whisper or the transcript, not the conversion).
  quality    predicted naturalness (UTMOS) >= MIN_NATURAL; drops conversion artifacts
             (E14: most clips 3.7-4.0, a few much lower)
  pitch      pitch-jump rate <= MAX_JUMPS %; drops glitchy melody
  length     0.8-12 s after trimming
Kept clips are trimmed, resampled to 22.05 kHz and written with a metadata.csv
("id|text", NO header: piper reads every line as a clip). Every decision is logged
to gate.csv so the selection can be audited and reproduced.

  .venv\\Scripts\\python distill\\make_dataset.py dailytalk_f1_seedvc_f0
"""

import csv
import sys
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf

from toast.naturalness import naturalness
from toast.pacing import trim_silence
from toast.text_normalize import normalize_for_speech
from toast.word_check import transcribe, word_errors

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "distill" / "data"
MIN_NATURAL = 3.5
MAX_JUMPS = 1.0
SAMPLE_RATE = 22050


def jump_rate(audio, sr):
    hz = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr).to_pitch_ac(
        time_step=0.01, pitch_floor=75, pitch_ceiling=500).selected_array["frequency"]
    voiced = hz > 0
    st = 12 * np.log2(np.where(voiced, hz, 1.0) / 100.0)
    both = voiced[1:] & voiced[:-1]
    return 100 * np.sum(np.abs(np.diff(st))[both] > 6) / max(voiced.sum(), 1)


def resample(audio, sr, target=SAMPLE_RATE):
    if sr == target:
        return audio
    t = np.linspace(0, len(audio) / sr, int(len(audio) * target / sr), endpoint=False)
    return np.interp(t, np.arange(len(audio)) / sr, audio).astype(np.float32)


def main(run):
    converted = DATA / "converted" / run
    out = DATA / "train" / run
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    with (DATA / "sources.csv").open(encoding="utf-8") as f:
        sources = {row["id"]: row for row in csv.DictReader(f, delimiter="|")}

    # Resume: clips checked in an earlier pass keep their decision (the gate is deterministic
    # given the same models), so the gate can run while conversion is still going.
    previous = {}
    if (out / "gate.csv").exists():
        with (out / "gate.csv").open(encoding="utf-8") as f:
            previous = {row["id"]: row for row in csv.DictReader(f, delimiter="|")}
    log, kept = [], []
    for n, path in enumerate(sorted(converted.glob("*.wav")), 1):
        clip_id = path.stem
        text = sources[clip_id]["text"]
        if clip_id in previous and (previous[clip_id]["kept"] != "True" or (out / "wavs" / path.name).exists()):
            row = previous[clip_id]
            row = {**row, "kept": row["kept"] == "True", "seconds": float(row["seconds"]),
                   "natural": float(row["natural"]), "jumps": float(row["jumps"]),
                   "wrong": int(row["wrong"]), "source_wrong": int(row["source_wrong"])}
            log.append(row)
            if row["kept"]:
                kept.append((clip_id, normalize_for_speech(text)))
            continue
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = trim_silence(audio.mean(axis=1), sr)
        seconds = len(audio) / sr
        wrong, _ = word_errors(text, transcribe(audio, sr))
        source_wrong = 0
        if wrong:
            src, src_sr = sf.read(DATA / "sources" / f"{clip_id}.wav", dtype="float32")
            source_wrong, _ = word_errors(text, transcribe(src, src_sr))
        natural = naturalness(audio, sr)
        jumps = jump_rate(audio, sr)
        reasons = []
        if wrong > source_wrong:
            reasons.append(f"words {wrong}>{source_wrong}")
        if natural < MIN_NATURAL:
            reasons.append(f"natural {natural:.2f}")
        if jumps > MAX_JUMPS:
            reasons.append(f"jumps {jumps:.2f}%")
        if not 0.8 <= seconds <= 12.0:
            reasons.append(f"length {seconds:.1f}s")
        log.append({"id": clip_id, "kept": not reasons, "reasons": "; ".join(reasons), "wrong": wrong,
                    "source_wrong": source_wrong, "natural": round(natural, 2), "jumps": round(jumps, 2),
                    "seconds": round(seconds, 2), "text": text})
        if not reasons:
            sf.write(out / "wavs" / f"{clip_id}.wav", resample(audio, sr), SAMPLE_RATE, subtype="PCM_16")
            kept.append((clip_id, normalize_for_speech(text)))
        if n % 100 == 0:
            print(f"  {n} checked, {len(kept)} kept", flush=True)

    with (out / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        for clip_id, text in kept:  # no header row
            f.write(f"{clip_id}|{text}\n")
    with (out / "gate.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log[0]), delimiter="|")
        writer.writeheader()
        writer.writerows(log)
    minutes = sum(r["seconds"] for r in log if r["kept"]) / 60
    dropped = [r for r in log if not r["kept"]]
    print(f"\nkept {len(kept)}/{len(log)} clips ({minutes:.1f} min); dropped {len(dropped)}")
    for reason in ("words", "natural", "jumps", "length"):
        print(f"  {reason:<8} {sum(reason in r['reasons'] for r in dropped)}")
    print(f"naturalness of kept clips: median {np.median([r['natural'] for r in log if r['kept']]):.2f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
