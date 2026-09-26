"""One consolidated CPU benchmark of the ToastTTS engine (default voice), for reports.

Measures, on this machine, in one sitting:
  load         time to load the voice and warm it up; process memory after loading
  full text    TTFA when the whole text is available (median of N), real-time factor
  streaming    TTFA from the FIRST LLM token + stalls, fake LLM at 30 and 10 tokens/s
  short input  TTFA for short single sentences (what say.py sees)
Appends a row per scenario to benchmarks/engine_benchmark.csv.

  .venv\\Scripts\\python scripts\\bench_engine.py
  .venv\\Scripts\\python scripts\\bench_engine.py --voice lessac --speed 1.0 --repeats 10
"""

import argparse
import csv
import ctypes
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from toast.engine import DEFAULT_SPEED, DEFAULT_VOICE
from toast.fake_llm import fake_llm_tokens
from toast.metrics import measure
from toast.pacing import stream_from_llm, stream_speech
from toast.voices import load_voice

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "engine_benchmark.csv"
SHORT = ["Well, hello there.", "Sure, I can help with that.", "Order #4521 ships at 3:15 PM.",
         "Is that right? Let me check.", "Thanks, have a great day!"]


def process_memory_mb():
    if platform.system() == "Windows":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = Counters()
        c.cb = ctypes.sizeof(c)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = ctypes.c_void_p  # a 64-bit handle; the default int return truncates it
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        if not get_info(get_process(), ctypes.byref(c), c.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return c.WorkingSetSize / 1e6
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    text = (ROOT / "samples" / "reference.txt").read_text(encoding="utf-8-sig").strip()

    memory_before = process_memory_mb()
    started = time.perf_counter()
    voice = load_voice(args.voice)
    voice.synthesize("Warm up.")
    load_s = time.perf_counter() - started
    memory_after = process_memory_mb()
    sr = voice.sample_rate

    rows = []

    def add(scenario, stats, note="", show_speed=True):
        # Streaming runs mostly wait for LLM tokens, so their audio/time ratio is not synthesis
        # speed and isn't reported.
        rows.append({"time": datetime.now().isoformat(timespec="seconds"), "voice": voice.name, "speed": args.speed,
                     "scenario": scenario, "ttfa_ms": round(stats["ttfa_ms"], 1), "stalls": stats.get("stalls", ""),
                     "stall_ms": round(stats["stall_ms"]) if "stall_ms" in stats else "",
                     "x_realtime": round(stats["x_realtime"], 1) if show_speed and "x_realtime" in stats else "",
                     "note": note, "memory_mb": round(memory_after), "load_s": round(load_s, 2)})

    _, s = measure(lambda: stream_speech(voice, text, speed=args.speed), sr, args.repeats)
    add("full text, 95-word paragraph", s)
    for tps in (30, 10):
        _, s = measure(lambda: stream_from_llm(voice, fake_llm_tokens(text, tps), speed=args.speed), sr, args.repeats)
        add(f"streaming from LLM, {tps} tokens/s (TTFA from first token)", s, show_speed=False)
    shorts = []
    for sentence in SHORT:
        _, s = measure(lambda: stream_speech(voice, sentence, speed=args.speed), sr, args.repeats)
        shorts.append(s["ttfa_ms"])
    add("short single sentences (median of 5 sentences)", {"ttfa_ms": float(np.median(shorts))},
        f"range {min(shorts):.0f}-{max(shorts):.0f} ms")

    cpu = platform.processor() or platform.machine()
    new = not OUT.exists()
    with OUT.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            writer.writeheader()
        writer.writerows(rows)

    print(f"Engine benchmark: {voice.name} at {args.speed}x, median of {args.repeats} runs, CPU only ({cpu}, "
          f"{os.cpu_count()} threads)\n")
    print(f"  load + warm-up        {load_s:.2f} s")
    print(f"  memory               {memory_after:.0f} MB process total ({memory_after - memory_before:+.0f} MB for the voice)")
    for r in rows:
        extra = (f", {r['stalls']} stalls" + (f" ({r['stall_ms']} ms total, worst run)" if r["stalls"] else "")
                 if r["stalls"] != "" else "")
        speed = f", {r['x_realtime']}x real time" if r["x_realtime"] else ""
        print(f"  {r['scenario']:<58} TTFA {r['ttfa_ms']:>6.1f} ms{extra}{speed} {r['note']}")
    print(f"\nSaved to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
