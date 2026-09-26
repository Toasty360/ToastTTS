#!/usr/bin/env python3
"""Key diagnostic for e18: do long-form chapters actually contain
non-punctuation pauses? Compares against e16's 99.2% punctuation figure.

Usage: python analyze_gaps.py --labels data/labels
"""
import argparse, json, os, statistics

PUNCT = set(",.!?;:")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--thr", type=float, default=150.0)
    args = ap.parse_args()
    tot_b, tot_p, p_punct, p_nonp = 0, 0, 0, 0
    nonp_gaps = []
    for name in ("train", "val"):
        fp = os.path.join(args.labels, f"examples_{name}.jsonl")
        nb, np_, npp, npn = 0, 0, 0, 0
        for line in open(fp):
            e = json.loads(line)
            for w, g in zip(e["words"][:-1], e["gaps_ms"]):
                nb += 1
                if g >= args.thr:
                    np_ += 1
                    if w and w[-1] in PUNCT:
                        npp += 1
                    else:
                        npn += 1
                        nonp_gaps.append(g)
        tot_b += nb; tot_p += np_; p_punct += npp; p_nonp += npn
        print(f"{name}: boundaries={nb} pauses(>={args.thr:.0f}ms)={np_} "
              f"({np_/max(nb,1)*100:.1f}%) after_punct={npp} ({npp/max(np_,1)*100:.1f}%) "
              f"non_punct={npn} ({npn/max(np_,1)*100:.1f}%)")
    nonp_gaps.sort()
    med = nonp_gaps[len(nonp_gaps)//2] if nonp_gaps else 0
    print(f"TOTAL: non-punctuation pauses = {p_nonp}/{tot_p} "
          f"({p_nonp/max(tot_p,1)*100:.1f}% of pauses), median {med:.0f}ms")
    print(f"e16 reference: 99.2% of pauses followed , or . "
          f"(i.e. ~0.8% non-punctuation)")

if __name__ == "__main__":
    main()
