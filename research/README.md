# ToastTTS: research notes

## Summary

Small on-device TTS models are fast but speak like they're reading a list: no pauses where a person would breathe, and a flat, even pace. ToastTTS is a streaming layer around such models (Piper, VITS) that:

- splits text at natural pause points as it arrives from an LLM;
- inserts punctuation-dependent pauses with smooth fades and continuous room tone;
- starts speaking before the reply is complete.

**Results** on one laptop CPU (i7-13700H, 2026-09-25):

- **Fast start:** speech begins **89–104 ms after the first LLM token** at 30 tokens/s (223 ms at 10 tokens/s), with no playback stalls in either case. Most of that time is spent waiting for the LLM, not synthesizing.
- **Voice choice:** 38 English Piper voices were benchmarked on naturalness (UTMOS), word errors (Whisper) and speed. Several match KittenTTS on predicted naturalness while starting **5–40× sooner**.
- **Cloud comparison:** on a replication of the Coval cloud TTS benchmark, scored with the same `whisper-1` transcriber, our voices reach **5.1–6.3% WER**. That places them among commercial services such as Cartesia Sonic (5.3–5.8%) and ElevenLabs Flash v2.5 (6.5%), at **75–97 ms TTFA locally** (cloud TTFA includes the network).
- **Pacing:** measured the same way, our pauses and (at 1.2×) speaking rate match reference recordings from Deepgram and Soniox.
- **The remaining gap is prosody:** word stress, pre-pause shaping and question intonation. A 15M-parameter model can't provide it through pacing alone ([08](notebook/08-open-questions.md)).

## Contents

| Document | What it covers |
|---|---|
| [methodology.md](methodology.md) | How every metric is measured, the environment, and known limits |
| [decisions.md](decisions.md) | Decision log with evidence, including reversed decisions |
| [distillation-plan.md](distillation-plan.md) | **Superseded:** the original Kokoro-teacher distillation design; replaced by human-speech sourcing ([14](notebook/14-human-source-and-converter.md)) and attempted in [15](notebook/15-training.md) |
| **Lab notebook** | |
| [01: Baseline](notebook/01-baseline.md) | Piper vs Kitten; fragment prosody, the core trade-off |
| [02: Natural pauses](notebook/02-natural-pauses.md) | Why the engine inserts its own pauses; the pacing design; two corrected claims |
| [03: Streaming](notebook/03-streaming.md) | Live chunker rules, TTFA, stalls, the "running low" rule |
| [04: Intelligibility](notebook/04-voice-intelligibility.md) | "It never said semicolon": the word check, and why the voices were changed |
| [05: Voice benchmark](notebook/05-voice-benchmark.md) | All 38 English Piper voices + Kitten |
| [06: Cloud comparison](notebook/06-coval-comparison.md) | Coval replication vs ElevenLabs, Deepgram, Cartesia and others |
| [07: Cloud pacing](notebook/07-cloud-reference-and-pacing.md) | Speaking rate and pauses vs Deepgram and Soniox; why amy at 1.2× |
| [08: Open questions](notebook/08-open-questions.md) | The prosody gap, options (Kokoro, distillation, learned pauses), unresolved measurements |
| [09: Dynamic input](notebook/09-dynamic-input.md) | First real-use test with `say.py`: device latency, leaning-word cuts, "#" normalization, monitor speakers swallowing words |
| [10: Kokoro](notebook/10-kokoro.md) | Kokoro-82M vs amy: pacing matches Soniox, UTMOS can't tell them apart, ~0.35 s start, a stall at 10 tokens/s |
| [11: Soft last word](notebook/11-phrase-final-softening.md) | Why Kokoro sounds natural (a soft last word before pauses), measuring it, adding it to amy |
| [12: Robotic melody and "Well,"](notebook/12-robotic-melody-and-well.md) | amy's pitch range is 2/3 of natural voices; pitch widening; why its "Well," sounds final; opener pauses |
| [13: VC feasibility](notebook/13-vc-feasibility.md) | Step 0 of distillation on Modal (~$0.04): kNN-VC keeps Kokoro's melody in amy's voice; open issue: 16 kHz bandwidth |
| [14: Human teacher, better converter](notebook/14-human-source-and-converter.md) | DailyTalk human speech replaces Kokoro; Seed-VC f0 (44.1 kHz) keeps 93% of the melody; quality gate; 1,023-clip training set |
| [15: Training](notebook/15-training.md) | Converting the training set, fine-tuning on Modal, infrastructure lessons, listening verdicts, voice-stability measures |
| [16: Word emphasis](notebook/16-word-emphasis.md) | The LLM marks the focus word, the engine performs it (pitch accent, lengthening, post-focus compression); subtle vs strong |
| [archive/](archive/) | The original, pre-measurement plan |

Supporting material: [experiments](../experiments/README.md) (re-runnable investigations with saved outputs) and [benchmark data](../benchmarks/README.md).

## Notes on rigor

- **Every number links to the file that produced it.** Anything measured only during exploration and not reproduced is labelled as such.
- **Corrections are recorded, not erased.** Two early single-run findings didn't hold up when repeated ([02](notebook/02-natural-pauses.md)).
- **Automatic metrics support listening; they don't replace it.** The final voice choice overrode the best WER (ryan) because it sounded less natural.
- **One machine, one listener.** Timings drift by 20–30 ms with machine state, and listening judgments are one person's. See [methodology](methodology.md) for all limits.
