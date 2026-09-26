#!/usr/bin/env python3
"""Build pause-label training examples from Whisper word-timestamp JSONs.

For each file: word sequence + per-boundary gap_ms = start[i+1]-end[i].
Writes examples_train.jsonl / examples_val.jsonl and gap_stats.json
(gap distribution, used to pick pause thresholds data-first).
"""
import argparse, json, os, glob, statistics

def load_examples(jsondir):
    exs = []
    for fp in sorted(glob.glob(os.path.join(jsondir, "*.json"))):
        d = json.load(open(fp))
        words = [w["w"] for w in d["words"]]
        if len(words) < 2:
            continue
        gaps = []
        for a, b in zip(d["words"][:-1], d["words"][1:]):
            gaps.append(max(0.0, (b["start"] - a["end"])) * 1000.0)
        exs.append({"file": d["file"], "words": words, "gaps_ms": gaps})
    return exs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--val-json", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tr = load_examples(args.train_json)
    va = load_examples(args.val_json)
    for name, exs in (("train", tr), ("val", va)):
        with open(os.path.join(args.outdir, f"examples_{name}.jsonl"), "w") as f:
            for e in exs:
                f.write(json.dumps(e) + "\n")

    all_gaps = [g for e in tr for g in e["gaps_ms"]]
    all_gaps.sort()
    n = len(all_gaps)
    def pct(p): return all_gaps[min(n - 1, int(n * p / 100))]
    pos = [g for g in all_gaps if g >= 150.0]
    stats = {
        "n_boundaries": n,
        "n_files": len(tr),
        "gap_ms_p50": pct(50), "gap_ms_p75": pct(75), "gap_ms_p90": pct(90),
        "gap_ms_p95": pct(95), "gap_ms_p99": pct(99), "gap_ms_max": all_gaps[-1] if n else 0,
        "frac_ge_150ms": len(pos) / max(n, 1),
        "frac_ge_300ms": sum(1 for g in all_gaps if g >= 300) / max(n, 1),
        "frac_ge_600ms": sum(1 for g in all_gaps if g >= 600) / max(n, 1),
        "mean_gap_ms": statistics.mean(all_gaps) if n else 0,
    }
    json.dump(stats, open(os.path.join(args.outdir, "gap_stats.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
