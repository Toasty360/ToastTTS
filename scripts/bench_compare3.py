#!/usr/bin/env python3
"""Head-to-head voice benchmark: ex02+clamp vs kokoro vs amy.

40-sentence eval set reproduced exactly like distill/evaluate_expresso.py:
30 held-out Expresso texts (gate.csv dropped rows, random seed 11)
+ 5 samples/reference.txt + 5 samples/word_test.txt sentences.

Each voice synthesizes in a FRESH SUBPROCESS so peak RSS is clean per voice.
The parent process then runs Whisper (faster-whisper small.en, int8 CPU) and
UTMOS once each across all clips.

Kokoro is reference-only: benchmark comparison is fine; never train on its output.

  .venv/Scripts/python scripts/bench_compare3.py

Writes:
  benchmarks/compare_ex02_kokoro_amy.csv
  out/listen/compare3/<voice>_sample.wav   (one sample clip per voice)
"""

import csv
import multiprocessing as mp
import random
import resource
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from toast.metrics import measure  # noqa: E402
from toast.naturalness import naturalness  # noqa: E402
from toast.pacing import clamp_pauses, stream_speech, trim_silence  # noqa: E402
from toast.splitter import split_into_sentences  # noqa: E402
from toast.text_normalize import normalize_for_speech  # noqa: E402
from toast.voices import load_voice  # noqa: E402
from toast.word_check import transcribe, word_errors, words  # noqa: E402

VOICES = [
    # (label, load_voice name, apply pause clamp, onnx path for size)
    ("ex02+clamp", "piper:en_US-amy_distill_ex02_stage1-best-medium@0.3/0.5",
     True, ROOT / "models" / "en_US-amy_distill_ex02_stage1-best-medium.onnx"),
    ("kokoro", "kokoro:af_heart",
     False, ROOT / "models" / "kokoro" / "kokoro-v1.0.onnx"),
    ("amy", "piper:en_US-amy-medium",
     False, ROOT / "models" / "en_US-amy-medium.onnx"),
]

SHORT_SENTENCE = "Hello, how are you today?"
SAMPLE_SENTENCE_IDX = 32  # "dead, digital silence" sentence (diagnostic)
N_REPEATS_TTFA = 3


def test_sentences():
    gate = ROOT / "distill" / "data" / "train" / "expresso_ex02" / "gate.csv"
    dropped = [r for r in csv.DictReader(gate.open(), delimiter="|") if r["kept"] != "True"]
    random.seed(11)
    heldout = [r["text"] for r in random.sample(dropped, min(30, len(dropped)))]
    ref = split_into_sentences((ROOT / "samples" / "reference.txt").read_text(encoding="utf-8-sig"))
    wt = split_into_sentences((ROOT / "samples" / "word_test.txt").read_text(encoding="utf-8-sig"))
    return heldout + ref + wt


def _worker(label, voice_name, do_clamp, clip_dir, conn):
    """Runs in a child process: synthesize all 40, report timing + peak RSS."""
    try:
        voice = load_voice(voice_name)
        sr = voice.sample_rate
        voice.synthesize("Warm up.")

        _, timing = measure(lambda: stream_speech(voice, SHORT_SENTENCE), sr,
                            repeats=N_REPEATS_TTFA)
        ttfa_ms = timing["ttfa_ms"]

        sentences = test_sentences()
        t0 = time.perf_counter()
        clips = []
        for s in sentences:
            y = voice.synthesize(normalize_for_speech(s))
            if do_clamp:
                y = clamp_pauses(y, sr, max_pause_ms=250)
            y = trim_silence(y, sr)
            clips.append(y.astype(np.float32))
        wall_s = time.perf_counter() - t0
        audio_s = sum(len(c) / sr for c in clips)

        out = Path(clip_dir)
        out.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(clips):
            sf.write(out / f"clip_{i:02d}.wav", c, sr)
        sf.write(out / "sample.wav", clips[SAMPLE_SENTENCE_IDX], sr)

        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        conn.send({"ok": True, "ttfa_ms": float(ttfa_ms),
                   "x_realtime": float(audio_s / wall_s),
                   "audio_s": float(audio_s), "wall_s": float(wall_s),
                   "peak_ram_mb": float(peak_mb), "sample_rate": sr,
                   "n_clips": len(clips)})
    except Exception:
        conn.send({"ok": False, "error": traceback.format_exc()})
    finally:
        conn.close()


def main():
    sentences = test_sentences()
    assert len(sentences) == 40, f"expected 40 sentences, got {len(sentences)}"
    words_checked = sum(len(words(s)) for s in sentences)
    print(f"eval set: {len(sentences)} sentences, {words_checked} normalized words")

    ctx = mp.get_context("spawn")
    results = {}
    with tempfile.TemporaryDirectory(prefix="bench3_") as tmp:
        for label, voice_name, do_clamp, onnx_path in VOICES:
            clip_dir = str(Path(tmp) / label)
            parent_conn, child_conn = ctx.Pipe(duplex=False)
            p = ctx.Process(target=_worker,
                            args=(label, voice_name, do_clamp, clip_dir, child_conn))
            t0 = time.perf_counter()
            p.start()
            child_conn.close()
            p.join(timeout=3600)
            dt = time.perf_counter() - t0
            if p.is_alive():
                p.terminate()
                results[label] = {"ok": False, "error": "worker timed out after 1h"}
                print(f"[{label}] TIMEOUT", flush=True)
                continue
            msg = parent_conn.recv() if parent_conn.poll(5) else {"ok": False, "error": "no reply"}
            msg["voice"] = label
            msg["wall_clock_s"] = dt
            msg["clip_dir"] = clip_dir
            msg["size_mb"] = onnx_path.stat().st_size / 1e6 if onnx_path.exists() else float("nan")
            results[label] = msg
            if msg["ok"]:
                print(f"[{label}] ttfa={msg['ttfa_ms']:.0f}ms x_rt={msg['x_realtime']:.1f} "
                      f"peak_ram={msg['peak_ram_mb']:.0f}MB size={msg['size_mb']:.1f}MB "
                      f"({dt:.0f}s wall)", flush=True)
            else:
                print(f"[{label}] FAILED: {msg['error'][-300:]}", flush=True)

        # Transcribe + UTMOS in the parent (models loaded once).
        rows = []
        listen_dir = ROOT / "out" / "listen" / "compare3"
        listen_dir.mkdir(parents=True, exist_ok=True)
        for label, voice_name, do_clamp, onnx_path in VOICES:
            r = results[label]
            if not r["ok"]:
                rows.append({"voice": label, "size_mb": f"{r['size_mb']:.1f}",
                             "peak_ram_mb": "", "ttfa_ms": "", "x_realtime": "",
                             "word_errors": "", "words_checked": words_checked,
                             "utmos": "", "note": "FAILED: " + r["error"][-200:]})
                continue
            sr = r["sample_rate"]
            clips = [sf.read(str(Path(r["clip_dir"]) / f"clip_{i:02d}.wav"),
                             dtype="float32")[0] for i in range(40)]
            errs, diffs = 0, []
            for s, c in zip(sentences, clips):
                heard = transcribe(c, sr)
                e, d = word_errors(s, heard)
                errs += e
                diffs.extend([(s[:40], x) for x in d])
            utmos = float(np.mean([naturalness(c, sr) for c in clips]))
            sf.write(listen_dir / f"{label}_sample.wav",
                     sf.read(str(Path(r["clip_dir"]) / "sample.wav"), dtype="float32")[0], sr)
            rows.append({"voice": label, "size_mb": f"{r['size_mb']:.1f}",
                         "peak_ram_mb": f"{r['peak_ram_mb']:.0f}",
                         "ttfa_ms": f"{r['ttfa_ms']:.0f}",
                         "x_realtime": f"{r['x_realtime']:.1f}",
                         "word_errors": errs, "words_checked": words_checked,
                         "utmos": f"{utmos:.2f}", "note": ""})
            print(f"[{label}] word_errors={errs}/{words_checked} utmos={utmos:.2f}", flush=True)
            if diffs:
                print(f"  diffs: {diffs[:8]}", flush=True)

    out_csv = ROOT / "benchmarks" / "compare_ex02_kokoro_amy.csv"
    cols = ["voice", "size_mb", "peak_ram_mb", "ttfa_ms", "x_realtime",
            "word_errors", "words_checked", "utmos", "note"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}")
    print(f"samples in {ROOT / 'out' / 'listen' / 'compare3'}")
    print("\n" + "\t".join(cols))
    for row in rows:
        print("\t".join(str(row[c]) for c in cols))


if __name__ == "__main__":
    main()
