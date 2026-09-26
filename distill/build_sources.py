"""Pick the DailyTalk clips that will teach amy natural delivery (laptop, free).

Source: DailyTalk (CC BY-SA 4.0), female speaker (speaker_id 1), from the
first parquet shard (models/dailytalk/). Held-out rule: conversations with
id % 10 == 0 are never used here; they are the evaluation set (E13/E14).

Filters (documented so the selection is reproducible):
  - 1.0-12.0 s long            (too short: little prosody; too long: GPU memory)
  - 3-40 words
  - plain text: letters, digits, basic punctuation only
  - not a near-duplicate of an earlier clip's text

Writes distill/data/sources/<id>.wav (original 24 kHz) and
distill/data/sources.csv (id|text|seconds|conversation|turn).

  .venv\\Scripts\\python distill\\build_sources.py
"""

import csv
import io
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from e13_vc_feasibility import DAILYTALK, DAILYTALK_FEMALE, is_heldout_conversation  # noqa: E402

OUT = ROOT / "distill" / "data"
ALLOWED = re.compile(r"^[A-Za-z0-9 ,.?!'\-:;$%]+$")


def main():
    rows = pq.read_table(DAILYTALK, columns=["conversation_id", "turn_id", "speaker_id", "text", "audio"]).to_pylist()
    (OUT / "sources").mkdir(parents=True, exist_ok=True)
    kept, seen, reasons = [], set(), {"held-out conversation": 0, "other speaker": 0, "length": 0,
                                       "word count": 0, "characters": 0, "duplicate": 0}
    for r in rows:
        if r["speaker_id"] != DAILYTALK_FEMALE:
            reasons["other speaker"] += 1
            continue
        if is_heldout_conversation(r["conversation_id"]):
            reasons["held-out conversation"] += 1
            continue
        text = " ".join(r["text"].split())
        audio, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        seconds = len(audio) / sr
        if not 1.0 <= seconds <= 12.0:
            reasons["length"] += 1
            continue
        if not 3 <= len(text.split()) <= 40:
            reasons["word count"] += 1
            continue
        if not ALLOWED.match(text):
            reasons["characters"] += 1
            continue
        key = re.sub(r"[^a-z]", "", text.lower())
        if key in seen:
            reasons["duplicate"] += 1
            continue
        seen.add(key)
        clip_id = f"dt{r['conversation_id']:05d}_{r['turn_id']:02d}"
        sf.write(OUT / "sources" / f"{clip_id}.wav", audio, sr)
        kept.append({"id": clip_id, "text": text, "seconds": round(seconds, 2),
                     "conversation": r["conversation_id"], "turn": r["turn_id"]})

    with (OUT / "sources.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "seconds", "conversation", "turn"], delimiter="|")
        writer.writeheader()
        writer.writerows(kept)
    total = sum(k["seconds"] for k in kept)
    questions = sum(k["text"].endswith("?") for k in kept)
    openers = sum(bool(re.match(r"^(Well|So|Oh|Okay|OK|Yes|Yeah|No|Actually|Sure|Right),", k["text"])) for k in kept)
    print(f"kept {len(kept)} clips, {total / 60:.1f} min (mean {total / max(len(kept), 1):.1f} s)")
    print(f"questions {100 * questions / len(kept):.0f}%, starting with an opener {100 * openers / len(kept):.0f}%")
    print("dropped:", reasons)


if __name__ == "__main__":
    main()
