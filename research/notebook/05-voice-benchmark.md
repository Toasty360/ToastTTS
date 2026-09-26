# 05: Benchmarking every English Piper voice

*2026-09-25 · Script: [`scripts/bench_voices.py`](../../scripts/bench_voices.py) · Data: [`benchmarks/voice_benchmark.csv`](../../benchmarks/voice_benchmark.csv) · Samples: `out/listen/all_voices/`*

## Question

Which Piper voice gives the best mix of naturalness, correct words and speed? And do any of them beat KittenTTS?

No published benchmark compares Piper voices with each other (commercial TTS services have published numbers), so the voices were measured directly.

## Setup

- All **38 English Piper voices** (US and UK, low/medium/high quality; 2.6 GB), plus **Kitten mini** as the reference to beat.
- Multi-speaker models (arctic, l2arctic, vctk, semaine, libritts, libritts_r) were tested with their **first speaker only**.
- Every voice goes through our pipeline (`smart` split, speed 0.9, each voice's default randomness):
  - **TTFA:** full text, median of 3 runs.
  - **Wrong words:** Whisper small.en on 3 recordings (reference paragraph × 2 + `word_test.txt`), 241 words in total.
  - **Naturalness:** UTMOS, mean over the 5 sentences of the paragraph.
- Whisper transcripts are saved in the CSV, so the word counts can be re-scored without new audio (`--rescore`).

## Measuring words fairly took four fixes

The first run counted many things as errors that weren't speaking mistakes. Each fix was verified with a test (`tests/test_word_check.py`) and the results were re-scored:

| Problem | Example | Affected |
|---|---|---|
| British spelling | "harbour" for *harbor* | all UK voices |
| Numbers written as words | "one hundred and eight" for *108* | cori, others |
| Joined compounds | "ultralowlatency" for *ultra-low latency* | many |
| Contractions | "is not" for *isn't* | Kitten went from 9 "errors" to 1 |

The first run also crashed at voice 9 because the CSV was open in Excel, which locks the file. The script now waits for the file to be released instead of crashing.

## Results

Sorted by naturalness; voices with more than 2 wrong words are listed after the rest ([full CSV](../../benchmarks/voice_benchmark.csv)):

| Voice | Natural | Wrong / 241 | TTFA | × real time |
|---|---|---|---|---|
| **en_US-lessac-high** | **4.44** | 0 | 184 ms | 4.1× |
| en_US-ryan-medium | 4.41 | 1 | 32 ms | 22.6× |
| **en_US-amy-medium** | **4.38** | 0 | 34 ms | 25.0× |
| Kitten mini (reference) | 4.35 | 1 | 1013 ms | 1.2× |
| en_US-lessac-medium | 4.32 | 0 | 42 ms | 21.7× |
| en_US-lessac-low | 4.29 | 1 | 38 ms | 27.8× |
| en_US-danny-low | 4.24 | 2 | 40 ms | 25.9× |
| en_US-bryce-medium | 4.22 | 0 | 32 ms | 26.1× |
| en_GB-jenny_dioco-medium | 4.21 | 0 | 30 ms | 23.8× |
| en_US-hfc_female-medium | 4.19 | 2 | 41 ms | 22.1× |
| en_US-kusal-medium | 4.16 | 2 | 37 ms | 19.5× |
| en_US-kathleen-low | 4.09 | 0 | 26 ms | 27.5× |
| en_GB-alba-medium | 4.06 | 2 | 35 ms | 22.1× |
| en_GB-alan-medium | 4.05 | 2 | 29 ms | 27.9× |
| en_GB-alan-low | 4.03 | 2 | 29 ms | 40.2× |
| en_US-joe-medium | 4.03 | 1 | 36 ms | 21.5× |
| *more than 2 wrong words:* | | | | |
| en_US-ryan-low | 4.34 | 4 | 24 ms | 29.1× |
| en_US-amy-low | 4.31 | 4 | 30 ms | 29.8× |
| en_US-hfc_male-medium | 4.31 | 4 | 36 ms | 21.1× |
| en_US-ryan-high | 4.27 | 10 | 113 ms | 4.5× |
| en_US-mike-medium | 4.23 | 5 | 38 ms | 20.8× |
| en_US-kristin-medium | 4.20 | 6 | 43 ms | 20.7× |
| en_US-ljspeech-high | 4.20 | 4 | 95 ms | 4.3× |
| en_US-libritts_r-medium | 4.15 | 3 | 33 ms | 23.1× |
| en_US-john-medium | 4.12 | 9 | 42 ms | 20.8× |
| en_GB-cori-high | 4.11 | 3 | 153 ms | 4.9× |
| en_US-sam-medium | 3.96 | 4 | 49 ms | 19.9× |
| en_GB-vctk-medium | 3.94 | 4 | 31 ms | 23.8× |
| en_US-norman-medium | 3.91 | 12 | 52 ms | 21.1× |
| en_GB-northern_english_male-medium | 3.90 | 4 | 37 ms | 22.9× |
| en_GB-southern_english_female-low | 3.90 | 20 | 33 ms | 29.4× |
| en_GB-semaine-medium | 3.88 | 8 | 48 ms | 25.5× |
| en_US-arctic-medium | 3.78 | 4 | 37 ms | 26.6× |
| en_GB-cori-medium | 3.76 | 36* | 37 ms | 22.3× |
| en_US-ljspeech-medium | 3.76 | 4 | 29 ms | 22.3× |
| en_US-libritts-high | 3.68 | 12 | 158 ms | 4.5× |
| en_GB-aru-medium | 3.55 | 13 | 36 ms | 23.7× |
| en_US-reza_ibrahim-medium | 3.45 | 9 | 69 ms | 17.4× |
| en_US-l2arctic-medium | 3.28 | 29 | 42 ms | 21.8× |

\* One cori-medium transcript is missing about 30 words at the end. It's unknown whether the voice or Whisper dropped them, so this score is unreliable.

## Findings

- **Several medium voices match or beat Kitten on predicted naturalness** (lessac-high, ryan, amy), and all Piper voices start **5–40× sooner**. On this laptop Kitten ran at only 1.2× real time in this run, which leaves almost no margin against stutter.
- **Differences under about 0.1 in naturalness are within noise.** Treat ryan, amy and lessac-medium as tied, and let listening decide.
- **Quality level isn't a reliable guide.** Low and medium voices often beat high ones (ryan-high: 10 wrong words; libritts-high: 3.68).
- **By ear** (the deciding test): ryan was rejected as unnatural despite its low error count; **amy was chosen** (see [07](07-cloud-reference-and-pacing.md)).

→ Next: [06: comparison with cloud providers](06-coval-comparison.md)
