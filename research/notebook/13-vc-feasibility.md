# 13: Step 0: can voice conversion carry Kokoro's delivery into amy's voice?

*2026-09-25 · Code: [`experiments/e13_vc_feasibility.py`](../../experiments/e13_vc_feasibility.py) (Modal), [`e13_evaluate.py`](../../experiments/e13_evaluate.py), [`e13_artifact_check.py`](../../experiments/e13_artifact_check.py) · Results: [evaluation](../../experiments/results/e13_vc_feasibility.txt), [artifacts](../../experiments/results/e13_artifact_check.txt) · Audio: `out/e13/20260925-2151/`, `out/listen/60_vc_*.wav`*

## Question

The distillation plan ([distillation-plan.md](../distillation-plan.md)) depends on one unproven link: if Kokoro speaks a sentence and a voice-conversion model re-voices it as amy, does Kokoro's melody survive, and does it sound like amy?

## Setup

- **Laptop (free):**
  - Kokoro `af_heart` spoke 10 test sentences: the 5 of `reference.txt` and the 5 of `word_test.txt`.
  - amy spoke the same 10 directly, as a baseline.
  - amy read about 9 minutes of her own original training sentences, as the reference for her voice.
- **Modal T4:** two MIT-licensed converters, one job each:
  - **kNN-VC** (WavLM-Large features + HiFi-GAN, 16 kHz output);
  - **OpenVoice v2** tone-color converter (22.05 kHz).
- **Laptop (free):** evaluation with our tools, plus **speaker similarity** (SpeechBrain ECAPA): does it sound like amy or like Kokoro?

**Credit hygiene** (following the author's Gaze-Detection project, where a failing final export once lost a run):
- A `--smoke` run on 1 clip goes first.
- Every clip is validated and written to a Modal Volume and committed the moment it exists.
- The laptop saves the audio before evaluating.
- `--fetch RUN_ID` re-downloads a run without the GPU; tested by deleting local copies and restoring them.

The smoke run caught a dead OpenVoice download link (404) for a few cents, and kNN-VC's result from that run was kept. **Total GPU time for the whole step: about 3.5 T4-minutes, about $0.04.**

## Results

| Condition | Pitch range 5–95% | Core range 10–90% | Pitch jumps | Similarity to amy | Similarity to Kokoro | Wrong words / 10 sentences | UTMOS | Energy > 7 kHz |
|---|---|---|---|---|---|---|---|---|
| Kokoro (source) | 9.80 st | 7.75 st | 0.25% | 0.48 | 0.92 | 0 | 4.31 | 1.78% |
| amy (direct) | 6.21 st | 4.98 st | 0.31% | 0.92 | 0.48 | 0 | 4.47 | 1.05% |
| **kNN-VC** | 18.73 st | **7.07 st** | **0.31%** | **0.75** | 0.51 | **0** | 4.39 | **0.21%** |
| OpenVoice v2 | 7.86 st | 6.21 st | 0.34% | 0.63 | 0.78 | 1 | 4.20 | 2.75% |

amy's own consistency (her clips compared with each other) is 0.84. kNN-VC's 0.75 is 89% of that.

"Well," in sentence 1 (pitch start / mid / end): Kokoro −1.2 / 0.0 / **+2.2** st (ends rising); amy +0.4 / 0.0 / **−1.5** (falls); **kNN-VC −0.5 / +0.1 / +1.8 (keeps Kokoro's rise)**; OpenVoice −0.9 / −0.1 / +1.4.

## Findings

- **kNN-VC passes Step 0 on every measured criterion:**
  - melody kept: a core range of 7.1 st, 91% of Kokoro's, where amy has 5.0;
  - no pitch glitches: the same jump rate as natural voices;
  - it sounds like amy: 0.75 similarity to amy vs 0.51 to Kokoro;
  - 0 wrong words, and naturalness 4.39.
- **The 18.7 st "range" was not real extra melody.** It came from a few extreme frames; the jump rate and the 10–90% range show the contour is clean. A useful lesson: **use percentile-trimmed range for converted audio.**
- **OpenVoice v2 fails:** it keeps more of Kokoro's voice than amy's (0.78 vs 0.63), loses melody (6.2 st core) and had 1 wrong word.
- **The open problem is bandwidth.** kNN-VC outputs 16 kHz: only 0.21% of energy above 7 kHz, against amy's 1.05%. amy's model trains at 22.05 kHz, so a student fine-tuned on this could learn a duller, muffled top end. **This must be solved or tested before Phase 1 training.** Options:
  1. **Mix the training data:** mostly VC clips for delivery, plus a share of amy's own full-band audio, so the model keeps its high frequencies.
  2. **Use a converter with 22/44 kHz output** (e.g. Seed-VC; check its license).
  3. **Bandwidth extension** of the VC output (e.g. audio super-resolution); extra cost and its own artifacts.

## Status

- **Step 0: passed on metrics with kNN-VC; listening pending** (`out/listen/60_vc_knnvc.wav` vs `60_vc_amy_direct.wav` and `60_vc_kokoro_source.wav`).
- The bandwidth question is added to the plan as **Step 0b**, to be settled before any training spend.

→ Plan: [distillation-plan.md](../distillation-plan.md)
