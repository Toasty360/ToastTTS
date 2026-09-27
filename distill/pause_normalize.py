"""Re-gate the ex02 training set with internal-pause normalization (local, free).

Stage 1 taught the model over-long pauses: ex02's dramatic readings contain
internal silences up to ~800 ms, and the model learned 440 ms comma pauses
from them. This rebuilds the training set with every internal silence longer
than CAP_MS shrunk to TARGET_MS, so stage 2 cannot re-learn the habit.

Only true silence is touched (breaths and room tone are louder than the
threshold); text is unchanged, so text<->audio alignment is preserved. Clips
are NOT split -- without word alignments we cannot split the text safely.

Writes distill/data/train/expresso_ex02_v2/ (wavs + metadata.csv, same ids
and texts) plus pause_report.json with the before/after internal-pause
distributions. VERIFY the report before any GPU spend: no internal pause
may exceed CAP_MS afterwards.

  .pause-venv/bin/python distill/pause_normalize.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toast.pacing import clamp_pauses, find_silences  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "distill" / "data" / "train" / "expresso_ex02"
DST = ROOT / "distill" / "data" / "train" / "expresso_ex02_v2"
CAP_MS = 300      # no internal pause may exceed this after normalization
TARGET_MS = 200   # long pauses are shrunk to this (natural comma range)
SAMPLE_RATE = 22050


def internal_pauses(audio, sr):
    """Durations (ms) of silences strictly inside the clip."""
    return [(e - s) / sr * 1000 for s, e in find_silences(audio, sr)
            if s > 0 and e < len(audio)]


def describe(pauses):
    a = np.array(pauses)
    if len(a) == 0:
        return {"n": 0}
    return {"n": int(len(a)), "p50_ms": round(float(np.median(a)), 1),
            "p95_ms": round(float(np.percentile(a, 95)), 1),
            "max_ms": round(float(a.max()), 1)}


def main():
    wavs = sorted((SRC / "wavs").glob("*.wav"))
    print(f"{len(wavs)} clips in {SRC / 'wavs'}", flush=True)
    texts = {}
    with (SRC / "metadata.csv").open(encoding="utf-8") as f:
        for line in f:
            clip_id, text = line.rstrip("\n").split("|", 1)
            texts[clip_id] = text

    (DST / "wavs").mkdir(parents=True, exist_ok=True)
    before, after = [], []
    per_clip, n_shrunk = [], 0
    kept_meta = []
    for n, wav in enumerate(wavs, 1):
        clip_id = wav.stem
        audio, sr = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:  # dataset is 22.05 kHz; be strict, don't guess
            raise SystemExit(f"{wav.name}: {sr} Hz, expected {SAMPLE_RATE}")
        pauses = internal_pauses(audio, sr)
        before.extend(pauses)
        new = clamp_pauses(audio, sr, max_pause_ms=TARGET_MS, min_pause_ms=CAP_MS)
        pauses2 = internal_pauses(new, sr)
        after.extend(pauses2)
        shrunk = len([p for p in pauses if p > CAP_MS])
        n_shrunk += shrunk
        per_clip.append({"id": clip_id, "pauses_before_ms": [round(p, 1) for p in pauses],
                         "pauses_after_ms": [round(p, 1) for p in pauses2]})
        sf.write(DST / "wavs" / wav.name, new, sr, subtype="PCM_16")
        kept_meta.append((clip_id, texts[clip_id]))
        if n % 200 == 0:
            print(f"  {n}/{len(wavs)} normalized", flush=True)

    with (DST / "metadata.csv").open("w", encoding="utf-8") as f:
        for clip_id, text in kept_meta:  # no header row: piper reads every line
            f.write(f"{clip_id}|{text}\n")

    report = {"clips": len(wavs), "cap_ms": CAP_MS, "target_ms": TARGET_MS,
              "clips_with_long_pause": sum(1 for c in per_clip if c["pauses_before_ms"]
                                           and max(c["pauses_before_ms"]) > CAP_MS),
              "pauses_shrunk": n_shrunk,
              "before": describe(before), "after": describe(after),
              "per_clip": per_clip}
    (DST / "pause_report.json").write_text(json.dumps(report, indent=1))
    print(f"\ninternal pauses BEFORE: {describe(before)}")
    print(f"internal pauses AFTER:  {describe(after)}")
    if report["after"].get("max_ms", 0) > CAP_MS:
        print("VERIFY FAILED: pauses above cap remain -- do not train", flush=True)
        sys.exit(1)
    print(f"VERIFY OK: max internal pause {report['after']['max_ms']} ms <= {CAP_MS} ms")
    print(f"-> {DST}")


if __name__ == "__main__":
    main()
