"""Measure how fast speech starts and whether playback would ever stutter.

We pretend to play the audio in real time while it is being made:
  - TTFA (time to first audio): from "here's the text" until the first
    sound could start playing. The model is already loaded and warmed up,
    like in a real app. Later, with a live LLM stream, TTFA will also
    include waiting for the first words to arrive.
  - stalls: times playback ran out of audio and had to wait for the next
    piece (heard as an unplanned gap). The goal is always 0.
"""

import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

RESULTS_FILE = Path(__file__).resolve().parent.parent / "benchmarks" / "results.csv"
COLUMNS = ["time", "label", "voice", "split", "speed", "ttfa_ms", "stalls", "stall_ms",
           "made_in_s", "audio_s", "x_realtime", "chunks", "word_errors"]


def run_once(chunks, sample_rate):
    """Pull every chunk from a speech stream, timing it like live playback."""
    start = time.perf_counter()
    audio, ttfa, play_end, stalls, stall_time, count = [], None, 0.0, 0, 0.0, 0
    for chunk in chunks:
        ready = time.perf_counter() - start
        if ttfa is None:
            ttfa = play_end = ready
        elif ready > play_end + 0.001:
            stalls += 1
            stall_time += ready - play_end
            play_end = ready
        play_end += len(chunk) / sample_rate
        audio.append(chunk)
        count += 1
    made_in = time.perf_counter() - start
    audio = np.concatenate(audio)
    return audio, {
        "ttfa_ms": ttfa * 1000,
        "stalls": stalls,
        "stall_ms": stall_time * 1000,
        "made_in_s": made_in,
        "audio_s": len(audio) / sample_rate,
        "x_realtime": len(audio) / sample_rate / made_in,
        "chunks": count,
    }


def measure(make_chunks, sample_rate, repeats=3):
    """Run several times; report the median TTFA and the worst stalls,
    because a single run can be unlucky (the laptop was busy)."""
    runs = [run_once(make_chunks(), sample_rate) for _ in range(repeats)]
    audio = runs[0][0]
    stats = runs[0][1].copy()
    stats["ttfa_ms"] = float(np.median([s["ttfa_ms"] for _, s in runs]))
    stats["made_in_s"] = float(np.median([s["made_in_s"] for _, s in runs]))
    stats["x_realtime"] = stats["audio_s"] / stats["made_in_s"]
    stats["stalls"] = max(s["stalls"] for _, s in runs)
    stats["stall_ms"] = max(s["stall_ms"] for _, s in runs)
    return audio, stats


def _upgrade_columns():
    """If the results file was written with fewer columns, rewrite it with
    the current ones (old rows get blanks), so every row lines up."""
    if not RESULTS_FILE.exists():
        return
    with RESULTS_FILE.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == COLUMNS:
            return
        rows = list(reader)
    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in COLUMNS} for row in rows)


def log_result(label, voice, split, speed, stats):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _upgrade_columns()
    new_file = not RESULTS_FILE.exists()
    with RESULTS_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "time": datetime.now().isoformat(timespec="seconds"),
            "label": label, "voice": voice, "split": split or "whole", "speed": speed,
            **{k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()},
        })
