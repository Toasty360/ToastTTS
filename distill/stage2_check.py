"""Stage-2 de-risk check: is the model un-learning the stretched pauses? (local)

Renders the 4 reference sentences with a candidate voice (no clamp -- we are
measuring what the MODEL learned, not the guardrail), writes listening clips
for Val's verdict, and measures comma-pause durations on 10 targeted
sentences.

Decision rule (baseline = ex02-stage1: comma p50=270ms, p95=420ms):
  shrinking        p95 clearly below 420 and trending down -> continue/extend
  shrinking slowly p95 in 300-420 -> extend run toward ~7h total
  not shrinking    p95 >= 420 -> abort, do not burn more GPU

  .pause-venv/bin/python distill/stage2_check.py "<voice>" <tag>

Clips -> ~/workspace/your_files/stage2_<tag>_1..4.wav (raw model, no clamp).
"""

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toast.pacing import find_silences, trim_silence  # noqa: E402
from toast.splitter import split_into_sentences  # noqa: E402
from toast.voices import load_voice  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pauses import COMMA_SENTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
YOUR_FILES = Path.home() / "workspace" / "your_files"
BASELINE_P95 = 420.0  # ex02-stage1 comma p95, ms


def internal_silences(audio, sr):
    return [(e - s) / sr * 1000 for s, e in find_silences(audio, sr)
            if s > 0 and e < len(audio)]


def main(voice_name, tag):
    ref = (ROOT / "samples" / "reference.txt").read_text(encoding="utf-8-sig")
    sents = split_into_sentences(ref)[:4]
    assert len(sents) == 4, f"expected 4 reference sentences, got {len(sents)}"
    print("reference sentences:", flush=True)
    for i, s in enumerate(sents, 1):
        print(f"  {i}. {s[:60]}...", flush=True)

    voice = load_voice(voice_name)
    voice.synthesize("Warm up.")
    sr = voice.sample_rate

    for i, s in enumerate(sents, 1):
        audio = trim_silence(voice.synthesize(s), sr)
        path = YOUR_FILES / f"stage2_{tag}_{i}.wav"
        sf.write(path, audio, sr)
        print(f"  clip {i}: {len(audio)/sr:.1f}s -> {path.name}", flush=True)

    # Targeted comma-pause measurement (10 sentences, no clamp).
    pauses = []
    for s in COMMA_SENTS[:10]:
        audio = trim_silence(voice.synthesize(s), sr)
        sil = internal_silences(audio, sr)
        if sil:
            pauses.append(max(sil))
    pauses = np.array(pauses)
    p50, p95, mx = np.median(pauses), np.percentile(pauses, 95), pauses.max()
    print(f"\ncomma pauses (n=10, no clamp): p50={p50:.0f}ms p95={p95:.0f}ms max={mx:.0f}ms", flush=True)
    print(f"baseline stage1: p50=270ms p95={BASELINE_P95:.0f}ms", flush=True)
    if p95 >= BASELINE_P95:
        print("VERDICT: NOT SHRINKING -> abort stage 2", flush=True)
    elif p95 > 300:
        print("VERDICT: SHRINKING SLOWLY -> extend run toward ~7h total", flush=True)
    else:
        print("VERDICT: SHRINKING -> continue", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: stage2_check.py <voice-name> <tag>")
    main(sys.argv[1], sys.argv[2])
