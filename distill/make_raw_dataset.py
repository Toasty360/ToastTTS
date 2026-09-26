"""Training set from the ORIGINAL human recordings (no voice conversion).

After hearing the converted clips, the author: "I don't really need amy's voice,
I just want a proper voice." Training directly on the human speaker gives her
real voice and delivery with no conversion artifacts; this is how Piper voices
are normally made (amy itself was fine-tuned from lessac on ~1 h of a new
speaker).

Uses distill/data/sources/ (DailyTalk speaker 1, training conversations only,
filters in build_sources.py): trims silence, resamples to 22.05 kHz, writes
metadata.csv ("id|text", no header).

  .venv\\Scripts\\python distill\\make_raw_dataset.py
"""

import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from toast.pacing import trim_silence
from toast.text_normalize import normalize_for_speech

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "distill" / "data"
NAME = "dailytalk_f1_raw"


def main():
    out = DATA / "train" / NAME
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    with (DATA / "sources.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="|"))
    total = 0.0
    with (out / "metadata.csv").open("w", encoding="utf-8", newline="") as meta:
        for row in rows:
            audio, sr = sf.read(DATA / "sources" / f"{row['id']}.wav", dtype="float32")
            audio = trim_silence(audio, sr)
            t = np.linspace(0, len(audio) / sr, int(len(audio) * 22050 / sr), endpoint=False)
            audio = np.interp(t, np.arange(len(audio)) / sr, audio).astype(np.float32)
            sf.write(out / "wavs" / f"{row['id']}.wav", audio, 22050, subtype="PCM_16")
            meta.write(f"{row['id']}|{normalize_for_speech(row['text'])}\n")
            total += len(audio) / 22050
    print(f"{NAME}: {len(rows)} clips, {total / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
