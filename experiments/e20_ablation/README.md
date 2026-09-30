# e20: Ablation — continued training on UNMODIFIED data

Question: did stage 2's pause collapse come from the pause normalization,
or just from 2h18m more training (convergence effect)?

Design: continue from the stage-1 checkpoint on the **unmodified**
`expresso_ex02` dataset — same hyperparameters, ~2.3h of new training time.
Only the data differs from stage 2 (which used the pause-normalized set).

If pauses stay long / get longer → normalization caused the stage-2 collapse.
If pauses shrink like stage 2 → continued training alone explains it.

## Runs

- Leg 1: `20260929-2323-from-20260926-2323-from-20260926-2143-from-20260926-2038`
  (from stage-1 `last.ckpt`, clock_offset_h=4.17, --hours 2.3)
  → 2,670 s new training, then L4 preemption (graceful SIGTERM stop).
- Leg 2: `20260930-0012-from-20260929-2323-from-20260926-2323-from-20260926-2038`
  (from leg-1 `last.ckpt`, clock_offset_h=4.91, --hours 1.56)
  → 7,017 s new training, finished cleanly.
- Total new training: 9,687 s = 2.69 h (target ~2.3 h; slight overshoot,
  which only strengthens the test).

## Best checkpoint

`en_US-amy_distill_20260930-0012-from-20260929-2323-from-20260926-2323-from-20260926-2143-from-20260926-2038_epoch_7124-val_mos_4.1909-medium.onnx`
(best val_mos 4.1909 of the exported set; ONNX + config JSON copied here)

Voice name for eval:
`piper:en_US-amy_distill_20260930-0012-from-20260929-2323-from-20260926-2323-from-20260926-2143-from-20260926-2038_epoch_7124-val_mos_4.1909-medium@0.3/0.5`

## Eval

`distill/eval_pauses.py` with the stage-2 protocol (40 targeted + 10 general
sentences, 3 repeats, noise_scale=0.3, noise_w=0.5, clamp on/off).
Results → `out/eval/pause_dist_<timestamp>.json` (copied here:
`pause_dist_20260930-0216.json`).

## Results (clamp off)

| voice | comma p50 | comma p95 | comma max | general p50 | general p95 | general max |
|---|---|---|---|---|---|---|
| amy (teacher) | 180 | 260 | 260 | 140 | 220 | 340 |
| ex02 stage 1 | 330 | 443 | 500 | 200 | 600 | 960 |
| ex02 stage 2 final | 140 | 205 | 300 | 140 | 260 | 280 |
| **ex02 ablation (unmodified, +2.69 h)** | **270** | **448** | **600** | **160** | **506** | **680** |

Reading: continued training on the unmodified data did **not** reproduce the
stage-2 collapse. The ablation voice sits much closer to stage 1 than to
stage 2 on every tail metric (comma p95 448 vs 205 ms; comma max 600 vs
300 ms; general p95 506 vs 260 ms; general max 680 vs 280 ms). There is a
small convergence effect (comma p50 330→270 ms, general max 960→680 ms),
but it does not explain stage 2's shift into the teacher's range. The pause
collapse is attributable to the normalization, not to more training.

(The synthesis-time clamp still bounds the ablation voice: clamp on →
comma max 260 ms, general max 260 ms.)

## Cost

- Leg 1: 2,670 s L4 (preempted, graceful stop)
- Leg 2: 7,017 s L4 training + ~5 min export on the same GPU
- Total: 9,687 s = 2.69 GPU-hours ≈ $2.15–$2.70 (Modal L4 $0.80/hr,
  up to ~$1.00/hr with regional multiplier)
- Budget: $10 approved; stage 2 used $3.96 → ablation brings the running
  total to roughly $6.40, inside budget.
