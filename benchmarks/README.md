# Benchmark data

Measured results, one file per benchmark. How each metric is defined: [research/methodology.md](../research/methodology.md). All data was produced on 2026-09-25 on the machine described there.

| File | Produced by | What it contains | Discussed in |
|---|---|---|---|
| [`voice_benchmark.csv`](voice_benchmark.csv) | `scripts/bench_voices.py` | All 38 English Piper voices + Kitten + 5 Kokoro voices (added later, [notebook/10](../research/notebook/10-kokoro.md)): TTFA, speed, wrong words, naturalness, what Whisper heard | [notebook/05](../research/notebook/05-voice-benchmark.md) |
| [`coval_tts_v1_whisper1.csv`](coval_tts_v1_whisper1.csv) | `scripts/bench_coval.py --asr whisper-1` | Coval tts-v1 replication scored by OpenAI whisper-1 (Coval's method): 4 voices × 30 prompts × 2 takes | [notebook/06](../research/notebook/06-coval-comparison.md) |
| [`coval_tts_v1.csv`](coval_tts_v1.csv) | `scripts/bench_coval.py` (local large-v2) | Same audio setup, scored by Whisper large-v2 on the laptop; TTFA copied from the whisper-1 re-timing so both files agree | [notebook/06](../research/notebook/06-coval-comparison.md) |
| [`voice_scan_en_US-libritts_r-medium.csv`](voice_scan_en_US-libritts_r-medium.csv) | `scripts/find_voices.py` | 60 of 904 libritts_r speakers on three tricky sentences; every one fails "semicolon" | [notebook/04](../research/notebook/04-voice-intelligibility.md) |
| [`engine_benchmark.csv`](engine_benchmark.csv) | `scripts/bench_engine.py` | One-sitting CPU benchmark of the engine (default voice): load time, memory, full-text TTFA, streaming TTFA and stalls at 30 and 10 tokens/s, short-sentence TTFA. Printed summary: [experiments/results/engine_benchmark_amy_1.2x.txt](../experiments/results/engine_benchmark_amy_1.2x.txt) | [README](../README.md) |
| [`results.csv`](results.csv) | `scripts/render.py`, `scripts/live.py` | Running log of every render/live run: a development history, including superseded voices and settings | all |

## Columns

**voice_benchmark.csv:** `voice`, `size_mb` (ONNX file), `ttfa_ms` (full text, median of 3), `x_realtime`, `wrong_words` / `words_checked` (Whisper small.en over 3 recordings, 241 words), `natural` (UTMOS, mean of 5 sentences), `mistakes` (expected→heard), `heard` (JSON list of the 3 transcripts, used by `--rescore`).

**coval_tts_v1*.csv:** one row per clip: `voice`, `id` (Coval test case A1–A30), `take`, `ttfa_ms` (Coval-style: first chunk + leading silence), `errors` (S+D+I after normalization), `ref_words`, `reference` / `heard` (both normalized). Pooled WER = Σ`errors` ÷ Σ`ref_words`.

**results.csv:** `time`, `label` (output file name), `voice`, `split`, `speed`, `ttfa_ms`, `stalls`, `stall_ms`, `made_in_s`, `audio_s`, `x_realtime`, `chunks` (speech + pause chunks), `word_errors` (only when `--check-words` was used). Early rows predate some columns and are blank there. TTFA is from the full text for render runs and from the first token for `live@…` runs.

## Caveats

- Timings are from one laptop and drift by about 20–30 ms with machine state. Compare only within a file.
- The power mode changed from "Balanced" to "Best performance" during the Coval run. The Coval TTFA values were re-measured together afterwards; `voice_benchmark.csv` timings are all from before the change.
- Cloud numbers in the Coval comparison come from the published leaderboard (2026-09-08), not from our own runs.
