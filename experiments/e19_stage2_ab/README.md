# e19_stage2_ab — blind A/B pick between the two stage-2 finalists

Stage 2 of the Expresso ex02 fine-tune produced two near-identical
finalist checkpoints that metrics cannot separate:

- **leg1**: leg-1 best, epoch 7059, val_mos 4.2126
- **leg2**: leg-2 best, epoch 7214, val_mos 4.1975

The standard 40-sentence eval ran on the leg-2 voice (11 word errors vs
amy's 12, UTMOS 4.25 vs 4.35, 0.74 similarity to ex02's recordings); the
leg-1 best was not separately evaluated because the validation scores
differ by 0.015. The pick is made by blind listening test.

## Fairness design

The only difference between the two files of a pair is the checkpoint.
For each sentence:

- same 10 sentences (the reference + word-test set used in every eval),
  covering commas, a semicolon, numbers, an opener comma, and
  subordinate-clause flow
- identical settings: noise_scale=0.3, noise_w=0.5, speed 1.0,
  ToastTTS "smart" split, default hand-written pause table, no clamp
- same seed per sentence, so the piece plan and pause-table draws are
  identical for both renders

A/B assignment is shuffled per sentence (seed 7). The mapping lives in
`wav/key.json`, which is not opened until the answers are in.

## Listening

In `wav/`: `01_A.wav` … `10_B.wav`, plus `sentences.txt` (the texts) and
`answers.txt` (one line per pair: `A`, `B`, or `same` — which sounds more
natural?).

Score with:

```bash
~/workspace/.pause-venv/bin/python ab.py --score
```

## Verdict

Pending listening. Not yet decided.
