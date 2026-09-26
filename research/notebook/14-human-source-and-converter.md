# 14: Human speech as the teacher, and a full-bandwidth converter

*2026-09-25 · Code: [`experiments/e13_vc_feasibility.py`](../../experiments/e13_vc_feasibility.py) (`--source dailytalk`), [`e13_evaluate.py`](../../experiments/e13_evaluate.py) · Results: [e14_vc_dailytalk.txt](../../experiments/results/e14_vc_dailytalk.txt) · Audio: `out/e13/20260925-2232-dailytalk/`, `out/listen/61_dt_*.wav`*

## Why change the teacher

Listening to E13 ([13](13-vc-feasibility.md)) raised three issues:

- **The converted voice sounded soft.** Measured: it was 27% darker than amy (spectral centroid 580 vs 795 Hz). The causes were kNN-VC's 16 kHz output and Kokoro's own dark tone bleeding through.
- **Kokoro's quirks were copied.** Kokoro says "dead," with a 9-semitone pitch swoop, and the converter faithfully copied it. **A student inherits its teacher's taste.**
- **Licensing.** Soniox's terms forbid using outputs *or derived data* to train or improve any speech-synthesis model ([D35](../decisions.md)).

**New teacher: real people.** DailyTalk is 20 hours of studio-recorded everyday conversation (CC BY-SA 4.0, 2,541 dialogues). We use speaker 1, the woman (median pitch 212 Hz vs 156 Hz for speaker 0). **Held-out rule:** conversations with `id % 10 == 0` are never used for training; evaluation clips come only from them.

## E14: three converters on 10 held-out DailyTalk clips

| Condition | Core pitch range (10–90%) | Pitch jumps | Energy > 7 kHz | Brightness | Similarity to amy | Similarity to source | Wrong words | UTMOS |
|---|---|---|---|---|---|---|---|---|
| DailyTalk source (human) | 10.03 st | 0.08% | 0.94% | 981 Hz | 0.18 | 0.79 | 1 | 4.19 |
| amy direct | 4.63 st | 0.21% | 1.10% | 803 Hz | 0.87 | 0.20 | 2 | 4.39 |
| kNN-VC (16 kHz) | 7.41 st | 0.35% | **0.13%** | 626 Hz | 0.64 | 0.30 | 3 | 4.21 |
| Seed-VC speech (22 kHz) | 7.06 st | 0.30% | 0.70% | 783 Hz | 0.69 | 0.25 | 2 | 3.99 |
| **Seed-VC f0-conditioned (44.1 kHz)** | **9.34 st** | **0.04%** | **0.60%** | 767 Hz | 0.62 | 0.28 | 2 | **3.78** |

amy's own consistency (her clips compared with each other) is 0.73.

GPU time: kNN-VC 30 s, Seed-VC 272 s, Seed-VC f0 332 s (T4, about $0.10 in total). A smoke run caught a crash loop first: Seed-VC's requirements pin an old `protobuf` that breaks Modal's client inside the container. It was fixed by upgrading protobuf after the requirements install.

**Criteria revised with justification:**
- **Melody** is now judged against *this* source (≥ 85% of its core range), using percentile-trimmed ranges.
- **Words:** amy herself mis-reads 2 DailyTalk words (names, casual speech), so the bar is "no more wrong words than amy's own reading".
- **Brightness:** energy above 7 kHz must be at least 50% of amy's.

**Verdict:**
- **Only Seed-VC f0 passes all six checks.** It keeps 93% of the human melody, has full bandwidth, the fewest pitch glitches, and is closer to amy than to the source.
- kNN-VC fails on melody (74%) and brightness. Seed-VC speech fails on melody (70%).

## Two problems found in Seed-VC's output

1. **A click at the very start.** Output begins at sample 0, and in clip 0 Whisper heard it as an extra word ("as"). Fix: a 15 ms fade-in in the conversion job.
2. **Lower predicted naturalness** (mean 3.78; per clip 2.85–4.00 vs the source's 3.87–4.41). Some clips carry conversion artifacts. Fix: **a quality gate** (`distill/make_dataset.py`). Every converted training clip must:
   - pass the word check (mismatches are forgiven only if the original human clip has the same count);
   - score UTMOS ≥ 3.5;
   - have ≤ 1% pitch jumps;
   - be 0.8–12 s long.

   Every decision is logged in `gate.csv`.

## Training source set

`distill/build_sources.py`, reproducible filters:
- DailyTalk speaker 1, first parquet shard, held-out conversations excluded;
- 1–12 s long, 3–40 words, plain text, no duplicates.

**1,023 clips, 58.9 minutes** (amy's original fine-tune used 1,019 clips, about 1 h). 36% are questions; 12% start with an opener like "Well,". Dropped: 121 held-out, 30 wrong length, 77 wrong word count, 64 unusual characters, 7 duplicates.

## Decisions

[D36](../decisions.md): human conversational speech (DailyTalk) replaces Kokoro as the delivery source; Seed-VC f0-conditioned replaces kNN-VC as the converter; a quality gate filters the converted training clips.

→ Next: [15: conversion and training](15-training.md)
