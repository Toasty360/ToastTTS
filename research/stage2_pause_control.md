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

Piper's duration predictor stays stochastic even at reduced noise. Two
identical runs of the pause eval (40 targeted sentences, `noise_scale=0.3`,
`noise_w=0.5`, single render per sentence) gave different tails for the
same stage-1 voice:

| run | stage-1 comma p50 | comma p95 | comma max | general max |
|---|---|---|---|---|
| 1 | 270 ms | 420 ms | 420 ms | 500 ms |
| 2 | 280 ms | 529 ms | 700 ms | 980 ms |

The median is stable (270→280 ms) but the tail swings by hundreds of
milliseconds run to run. Rendering each sentence 3× and taking the median
(targeted) or pooling (general) removes the tail noise; all tables below
use the 3-repeat protocol. The residual lesson: at `noise_w=0.5` a single
render's longest pause is not a reliable measurement.

## Experiment 2: pause-normalized fine-tuning

Internal silences longer than 300 ms in the 1,546 gated ex02 clips were
shortened to 200 ms (audio-domain edit; text unchanged). Distribution shift:

| split | n pauses | p50 | p95 | max |
|---|---|---|---|---|
| before | 6,521 | 60 ms | 420 ms | 1,820 ms |
| after | 6,534 | 60 ms | 207 ms | 300 ms |

Stage 2 fine-tunes the stage-1 checkpoint on this normalized set. (Methods
note: stage 2 resumed from the stage-1 run's `last.ckpt`, not the
epoch-7004 best checkpoint, `val_mos` 4.1634. The epoch-7004 checkpoint
remains on the training volume.)

Training ran in two legs on a Modal L4 GPU, 2 h 18 min of new training
time total (leg 1: 39 min, reaching epoch 7119; leg 2: 100 min, reaching
epoch 7224). Two service restarts killed the local Modal client mid-run,
but the remote GPU jobs and volume checkpoints survived; training was
resumed from the persisted checkpoints each time. Lightning's `max_time`
counts cumulative time stored in resumed checkpoints, so a `--clock-offset`
flag was added to make `--hours` mean new training time.

Pause distributions measured on 40 targeted sentences (20 comma / 10
semicolon / 10 colon) plus 10 general sentences, 3 renders each at
`noise_scale=0.3`, `noise_w=0.5`:

| voice | comma p50 | comma p95 | comma max | general p50 | general p95 | general max |
|---|---|---|---|---|---|---|
| amy (teacher) | 180 | 260 | 260 | 140 | 220 | 340 |
| ex02 stage 1 | 330 | 443 | 500 | 200 | 600 | 960 |
| ex02 stage 2 final | 140 | 205 | 300 | 140 | 260 | 280 |
| libritts_r-medium spk 3922 | 45 | 80 | 80 | 120 | 310 | 400 |

Reading: stage 1 learned comma pauses roughly twice the teacher's
(p50 330 vs 180 ms; p95 443 vs 260 ms) and produced general-distribution
pauses up to ~1 s. After stage 2 on the pause-normalized set, the comma
distribution sits inside the teacher's range (p50 140 vs 180 ms;
p95 205 vs 260 ms; max 300 vs 260 ms), and the general-distribution tail
collapsed (p95 260 vs 600 ms; max 280 vs 960 ms). The stock Piper voice
barely pauses at commas at all (p50 45 ms).

Limitation: the targeted punctuation rows use the longest internal silence
per sentence as a proxy for the mark's pause, which can select an
unrelated pause; the general-distribution rows (all internal silences
≥100 ms) do not have this problem and tell the same story.

## Standard voice metrics (stage-2 final)

40 sentences (30 held-out Expresso gate texts + reference/word-test),
Whisper `small.en` for word errors, UTMOS for naturalness, ECAPA speaker
embeddings for voice similarity:

| metric | amy | ex02 stage 2 final |
|---|---|---|
| word errors (lower better) | 12 | 11 |
| UTMOS naturalness | 4.35 | 4.25 |
| similarity to real ex02 | 0.25 | 0.74 |
| similarity to amy | 0.81 | 0.30 |
| time to first audio | 144 ms | 107 ms |
| throughput | 11.3× realtime | 13.3× realtime |

All six pre-registered checks pass: no more word errors than amy, UTMOS
within 0.3 of amy (breakage alarm only), one clean voice (0.74 similar to
ex02's own recordings vs her own consistency 0.48; 0.30 to amy vs 0.37
ceiling), TTFA and speed within 10% of amy. Versus stage 1 (17 word
errors, UTMOS 4.01), stage 2 is cleaner on both counts while keeping the
voice identity (0.74 vs 0.75).

## Experiment 3: synthesis-time pause clamp (opt-in)

`toast/pacing.py` gained an opt-in `max_pause_ms`: internal silences longer
than the cap are shortened by removing middle frames with a 3 ms crossfade.
Leading/trailing silence and inter-piece handwritten pauses are untouched;
the clamp applies within synthesized pieces only. Effect on stage 1 (cap
250 ms, 3-repeat protocol):

| voice | comma p95 | comma max | general p95 | general max |
|---|---|---|---|---|
| ex02 stage 1, clamp off | 443 | 500 | 600 | 960 |
| ex02 stage 1, clamp on | 260 | 260 | 260 | 280 |

The clamp is a hard upper bound on intra-piece pauses; it cannot fix pauses
the model never produces (libritts_r's 45 ms commas are untouched), and
inter-piece pauses set by the handwritten pacing table remain as written.

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

Stage-2 training complete (2 h 18 min new GPU time, 2 legs). Pause
distributions and standard metrics measured; synthesis-time clamp verified.
Listening clips rendered at every export point
(`your_files/stage2_mid_*.wav`, `your_files/stage2_final_*.wav`);
listening is the final quality gate. Not started: stage 3.

Checkpoints preserved: Modal volume `toasttts-train` (full run histories)
plus local verified copies of the leg-1 best (epoch 7059, val_mos 4.2126),
leg-2 best (epoch 7214, val_mos 4.1975), and leg-2 last checkpoint.
