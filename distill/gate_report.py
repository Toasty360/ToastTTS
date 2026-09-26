"""Summarise the quality gate's decisions (reads gate.csv; no recomputation).

  .venv\\Scripts\\python distill\\gate_report.py dailytalk_f1_seedvc_f0
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(run):
    rows = list(csv.DictReader((ROOT / "distill" / "data" / "train" / run / "gate.csv").open(encoding="utf-8"),
                               delimiter="|"))
    kept = [r for r in rows if r["kept"] == "True"]
    natural = np.array([float(r["natural"]) for r in rows])
    jumps = np.array([float(r["jumps"]) for r in rows])
    seconds = np.array([float(r["seconds"]) for r in rows])
    print(f"clips checked {len(rows)}, kept {len(kept)} ({100 * len(kept) / len(rows):.0f}%), "
          f"{sum(float(r['seconds']) for r in kept) / 60:.1f} of {seconds.sum() / 60:.1f} min kept")
    print(f"\nbreakage alarm (UTMOS) over all converted clips: p10 {np.percentile(natural, 10):.2f}, "
          f"median {np.median(natural):.2f}, p90 {np.percentile(natural, 90):.2f}; below 3.5: {(natural < 3.5).sum()}")
    print(f"pitch jumps: median {np.median(jumps):.2f}%, above 1%: {(jumps > 1).sum()}")
    worse = [r for r in rows if int(r["wrong"]) > int(r["source_wrong"])]
    forgiven = [r for r in rows if 0 < int(r["wrong"]) <= int(r["source_wrong"])]
    print(f"word check: {len(worse)} clips worse than the original human clip (dropped); "
          f"{len(forgiven)} mismatches forgiven (the human clip had the same count)")
    for r in worse[:5]:
        print(f"   e.g. {r['id']}: {r['text'][:70]!r} (wrong {r['wrong']} vs human {r['source_wrong']})")
    print("\ndrop reasons (a clip can have several):",
          {k: sum(k in r["reasons"] for r in rows) for k in ("words", "natural", "jumps", "length")})
    only = {k: sum(r["reasons"].startswith(k) and ";" not in r["reasons"] for r in rows)
            for k in ("words", "natural", "jumps", "length")}
    print("dropped for that reason alone:", only)
    q = sum(r["text"].endswith("?") for r in kept)
    o = sum(r["text"].split(",")[0].lower() in ("well", "so", "oh", "okay", "yes", "yeah", "no", "actually", "sure")
            for r in kept)
    print(f"\nkept set: {100 * q / len(kept):.0f}% questions, {100 * o / len(kept):.0f}% start with an opener, "
          f"mean {np.mean([float(r['seconds']) for r in kept]):.1f} s")


if __name__ == "__main__":
    main(sys.argv[1])
