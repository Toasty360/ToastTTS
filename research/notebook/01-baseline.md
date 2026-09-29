# 01: Baseline: which small model, and does it survive being cut up?

*2026-09-25 · Experiment [E01](../../experiments/e01_compare_piper_kitten.py)*

## Question

The original plan ([archive/original-plan.md](../archive/original-plan.md)) was written for a Raspberry Pi with targets set in advance. The project's actual target is a **laptop CPU**, and the core idea is simple: speech should pause where a person would pause, instead of running all the words together. Before building anything, a base model was needed that:

1. runs fast on a CPU, and
2. still sounds acceptable when fed short pieces of text, because streaming means speaking before the whole reply exists.

## Setup

- **Piper** (VITS, ONNX): `en_US-lessac-medium`, and later speaker 3922 of `en_US-libritts_r-medium`, the voice used in the InterviewAgent app.
- **KittenTTS 0.8.1**: `mini` (80M parameters) and `micro` (40M).
- **Environment:** KittenTTS 0.8.1 depends on `misaki`, which requires Python < 3.13, so the project runs on Python 3.12 via `uv`. KittenTTS also pulls in PyTorch.

E01 renders `samples/reference.txt` two ways per model: **whole** (one call) and **pieces** (split at every punctuation mark, glued back to back with nothing added).

## Results

| Model | Load | Whole paragraph | First piece ready | × real time |
|---|---|---|---|---|
| Piper lessac-medium | 1.5 s | 1.14 s | 31 ms | 30.7× |
| Piper libritts_r 3922 | 1.5 s | 0.80 s | 20 ms | 33.0× |
| Kitten micro | 1.6 s | 11.48 s | 377 ms | 3.6× |
| Kitten mini | 9.1 s | 19.93 s | 703 ms | 1.9× |

- **Kitten pads every piece with silence.** Its "pieces" recordings were about 10 s longer than "whole" (48.8 s vs 38.5 s for mini). Piper's were almost the same length (35.8 s vs 34.9 s).
- **Listening:** Kitten *whole* was judged good. Kitten *pieces* was heard as choppy.

## Findings

- **Fragment prosody is the central problem.** A model given "12," on its own speaks it like a complete statement: pitch falls, delivery sounds final, then resets for the next piece. Given the whole sentence, it plans intonation across all of it. So cutting text finely gives a fast start but hurts naturalness. Every later design decision trades these two off.
- **Piper is 15–30× faster than Kitten on this CPU.** That speed is what makes a fast start possible.
- **The model matters less than the layer around it.** Chunking, pauses and fades are model-agnostic, so all models sit behind one interface (`toast/voices.py`: text + speed → float32 audio + sample rate), and the rest of the code never assumes a sample rate (Kitten uses 24 kHz, Piper 22.05 kHz).

→ Next: [02: natural pauses](02-natural-pauses.md)
