# Controlling pauses in VITS-based TTS

Research note — pause behavior in Piper fine-tuning and at synthesis time.
Dataset: Expresso (CC BY-NC 4.0), speaker ex02. Student: Piper medium
(~63 MB ONNX), distilled from the `en_US-amy-medium` voice.

## Background

Piper (a VITS model) learns pause durations from its training data through a
stochastic duration predictor. Two pause problems showed up when fine-tuning
on Expresso speaker ex02, a dramatic reader:

1. **Random phantom pauses.** At the default `noise_w=0.8`, rendering the
   same sentence twice produced different pause placements. Lowering the
   duration-noise knob to `noise_w=0.5` with `noise_scale=0.3` removed the
   phantom pauses with no retraining (inference-only change).
2. **Learned over-long pauses.** The stage-1 fine-tune reproduced ex02's
   dramatic style: comma pauses stretched far beyond the teacher voice.

## Experiment 1: noise_w quantification

> *Pending: render N sentences K times each at noise_w 0.8 / 0.5 / 0.3 and
> measure pause-placement variance.*

## Experiment 2: pause-normalized fine-tuning

Internal silences longer than 300 ms in the 1,546 gated ex02 clips were
shortened to 200 ms (audio-domain edit; text unchanged). Distribution shift:

| split | n pauses | p50 | p95 | max |
|---|---|---|---|---|
| before | 6,521 | 60 ms | 420 ms | 1,820 ms |
| after | 6,534 | 60 ms | 207 ms | 300 ms |

Stage 2 fine-tunes the stage-1 best checkpoint on this normalized set.

Pause distributions measured on 40 targeted sentences (20 comma / 10
semicolon / 10 colon) plus 40 general sentences, all at `noise_scale=0.3`,
`noise_w=0.5`:

| voice | comma p50 | comma p95 | comma max | general p95 | general max |
|---|---|---|---|---|---|
| amy (teacher) | 180 | 260 | 260 | 199 | 360 |
| ex02 stage 1 | 270 | 420 | 420 | 442 | 500 |
| ex02 stage 2, leg 1 (~0.65 h) | 140 | 226 | 340 | — | — |
| libritts_r-medium spk 3922 | 40 | 85 | 100 | 309 | 320 |

Reading: stage 1 learned comma pauses ~1.5x longer than the teacher
(p50 270 vs 180 ms; max 420 vs 260 ms). After ~0.65 h of stage-2 training on
the pause-normalized set, comma pauses collapsed to p50 140 / p95 226 ms —
below the teacher's p50, suggesting the normalization may even be
over-correcting the median. The stock Piper voice barely pauses
at commas at all (p50 40 ms). *[Stage-2 final row pending.]*

## Experiment 3: synthesis-time pause clamp (opt-in)

`toast/pacing.py` gained an opt-in `max_pause_ms`: internal silences longer
than the cap are shortened by removing middle frames with a 3 ms crossfade.
Leading/trailing silence and inter-piece handwritten pauses are untouched.
Effect on stage 1 (cap 250 ms):

| voice | comma p95 | comma max | general max |
|---|---|---|---|
| ex02 stage 1, clamp off | 420 | 420 | 500 |
| ex02 stage 1, clamp on | 260 | 260 | 260 |

The clamp is a hard upper bound; it cannot fix pauses the model never
produces (libritts_r's 40 ms commas are untouched).

## Honest negatives

- A 93.6k-param BiLSTM pause model (F1 0.934) mostly learned
  punctuation→pause (99.2% of training pauses follow `,` or `.`).
- A long-form retrain on 5.56 h of audiobooks failed cleanly (F1 0.354):
  only 1.4% of pauses were not after punctuation, so there was no
  phrase-break signal to learn.
- Long-form data did not buy word correctness: the stock
  `libritts_r-medium` voice renders "semicolon" as "Sima"; the ex02 and amy
  voices render it correctly.

## Status

Stage-2 training in progress on the pause-normalized set. De-risk check at
~1.5–2 h: export a preview, measure comma pauses; abort if not shrinking,
extend up to ~7 h total if shrinking slowly.
