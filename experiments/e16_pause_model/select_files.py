#!/usr/bin/env python3
"""Select a subset of LibriSpeech flac files for pause-model training.

Strategy: longest files first (more words/boundaries per Whisper-minute),
then split speakers into train/val (speaker-disjoint, no leakage).
Writes: files_train.txt, files_val.txt (one flac path per line).
"""
import argparse, os, subprocess, collections

def duration(fp):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", fp], capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="LibriSpeech/dev-clean dir")
    ap.add_argument("--target-sec", type=float, default=3600)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--val-speaker-frac", type=float, default=0.2)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    files = []
    for dp, _, fns in os.walk(args.root):
        for fn in fns:
            if fn.endswith(".flac"):
                files.append(os.path.join(dp, fn))
    print(f"found {len(files)} flac files, probing durations...")
    recs = [(fp, duration(fp)) for fp in files]
    recs = [(fp, d) for fp, d in recs if d > 0]
    recs.sort(key=lambda r: -r[1])

    # longest-first until target seconds
    sel, total = [], 0.0
    for fp, d in recs:
        sel.append((fp, d)); total += d
        if total >= args.target_sec:
            break
    print(f"selected {len(sel)} files = {total/3600:.2f}h")

    # speaker = first path component under root
    def speaker(fp):
        return os.path.relpath(fp, args.root).split(os.sep)[0]
    by_spk = collections.defaultdict(list)
    for fp, d in sel:
        by_spk[speaker(fp)].append((fp, d))
    spks = sorted(by_spk, key=lambda s: -sum(d for _, d in by_spk[s]))
    n_val = max(1, int(len(spks) * args.val_speaker_frac))
    val_spks = set(spks[:n_val])
    print(f"speakers: {len(spks)} total, {len(val_spks)} held out for val: {sorted(val_spks)}")

    with open(os.path.join(args.outdir, "files_train.txt"), "w") as f:
        for fp, d in sel:
            if speaker(fp) not in val_spks:
                f.write(fp + "\n")
    with open(os.path.join(args.outdir, "files_val.txt"), "w") as f:
        for fp, d in sel:
            if speaker(fp) in val_spks:
                f.write(fp + "\n")
    tr = sum(d for fp, d in sel if speaker(fp) not in val_spks)
    va = sum(d for fp, d in sel if speaker(fp) in val_spks)
    print(f"train: {tr/60:.1f} min, val: {va/60:.1f} min")

if __name__ == "__main__":
    main()
