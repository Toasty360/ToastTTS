#!/usr/bin/env python3
"""Transcribe flac files with local Whisper small.en, saving word timestamps.

Usage:
    python transcribe.py --files data/subset1h/files_train.txt --outdir data/words_train
CPU-only. Prints per-file timing incl. realtime factor for planning.
Skips files whose output JSON already exists (safe to resume).
"""
import argparse, json, os, time
import whisper

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", default="small.en")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"loading whisper {args.model} on CPU...", flush=True)
    model = whisper.load_model(args.model, device="cpu")
    files = [l.strip() for l in open(args.files) if l.strip()]
    print(f"{len(files)} files", flush=True)

    tot_audio, tot_wall = 0.0, 0.0
    for i, fp in enumerate(files):
        out = os.path.join(args.outdir,
                           os.path.basename(fp).replace(".flac", ".json"))
        if os.path.exists(out):
            continue
        t0 = time.time()
        try:
            result = model.transcribe(fp, word_timestamps=True,
                                      language="en", verbose=False)
        except Exception as e:
            print(f"[{i+1}/{len(files)}] ERROR {fp}: {e}", flush=True)
            continue
        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                txt = w["word"].strip()
                if txt:
                    words.append({"w": txt, "start": round(w["start"], 3),
                                  "end": round(w["end"], 3)})
        json.dump({"file": fp, "words": words}, open(out, "w"))
        dt = time.time() - t0
        dur = words[-1]["end"] if words else 0.0
        tot_audio += dur; tot_wall += dt
        rtf = dt / max(dur, 0.01)
        print(f"[{i+1}/{len(files)}] {os.path.basename(fp)} "
              f"words={len(words)} audio={dur:.1f}s wall={dt:.1f}s rtf={rtf:.2f}x",
              flush=True)
    if tot_audio > 0:
        print(f"DONE: {tot_audio/60:.1f} min audio in {tot_wall/3600:.2f}h "
              f"(avg rtf={tot_wall/tot_audio:.2f}x)")

if __name__ == "__main__":
    main()
