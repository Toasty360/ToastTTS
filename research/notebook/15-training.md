# 15: Converting the training set and fine-tuning amy (Phase 1)

*2026-09-25 · Code: [`distill/build_sources.py`](../../distill/build_sources.py), [`distill/convert_modal.py`](../../distill/convert_modal.py), [`distill/make_dataset.py`](../../distill/make_dataset.py), [`distill/train_modal.py`](../../distill/train_modal.py), [`distill/evaluate_student.py`](../../distill/evaluate_student.py)*

*Status: in progress. Results are filled in as they arrive.*

## Pipeline

```
DailyTalk speaker 1 (training conversations only)      laptop   build_sources.py   1,023 clips, 58.9 min
   └─ Seed-VC f0 (44.1 kHz) → amy's voice               Modal T4 convert_modal.py   4 parallel containers
        └─ quality gate (words, UTMOS ≥ 3.5, jumps, len) laptop   make_dataset.py    gate.csv logs every decision
             └─ fine-tune amy-medium (piper1-gpl)        Modal L4 train_modal.py     capped hours, checkpoints → Volume
                  └─ ONNX export + speak test            Modal CPU                   separate step, re-runnable
                       └─ held-out evaluation            laptop   evaluate_student.py criteria fixed in advance
```

## Infrastructure lessons (all caught by smoke runs, total ≈ $0.10)

Each would have cost a full run if found late:

| Problem | Where | Fix |
|---|---|---|
| Seed-VC's requirements pin an old `protobuf`, which breaks Modal's own client: the container crash-loops at startup | conversion | Upgrade protobuf after the requirements install. The laptop waits at most 30 min per job. The crash-looping app was stopped by hand |
| `python setup.py build_ext` needs `scikit-build`, which only exists inside pip's isolated build | training image | Install scikit-build, cmake, ninja and cython explicitly (failed at build time, no GPU cost) |
| PyTorch's strict `weights_only` loader rejects amy's 2023 checkpoint (it pickles `pathlib.PosixPath`) | training | Load in full mode for this trusted checkpoint (official `rhasspy/piper-checkpoints`) |
| Newer Lightning CLIs turn a checkpoint's stored hyper-parameters into command-line options, and amy's are from an older trainer (`sample_bytes`, …) | training | Clean copy of the checkpoint: weights and training state kept, `hyper_parameters` dropped |
| Piper trains forever by default (`max_epochs: -1`) | training | Hard `--trainer.max_time`; the laptop refuses runs over 7.5 h |
| A local command timeout would stop an attached Modal app | all | `modal run --detach`: the job continues if the laptop disconnects |
| Seed-VC's reference clip is a prompt in every diffusion step: 25 s of reference makes each ~3.5 s conversion ~40 s on a T4, and sources over 5 s run in several passes | conversion | **Not changed mid-run:** E14 validated the 25 s setting, and switching would mix settings or discard finished clips. **For future runs: validate a shorter (~10 s) reference first** |

**Safeguards verified working:**
- A re-run skipped all 3 finished clips in 0 s (resume).
- `--fetch` restored deleted local results from the Volume.
- Export produced a 63.5 MB model (amy: 63 MB, the same architecture, as D32 requires) that loads in `toast` and speaks: 59 ms TTFA, 0 wrong words.

## Training smoke run (plumbing test, not a real student)

10 minutes on an L4, 40 **raw** DailyTalk clips (not converted), 90 epochs, 11 checkpoints saved.

The evaluation (`evaluate_student.py`, held-out text) behaved as expected for a model pulled toward a *different speaker*:

| | Core pitch range | UTMOS | Wrong words / 40 sentences | Similarity to amy | Similarity to the human | TTFA |
|---|---|---|---|---|---|---|
| amy | 4.72 st | 4.41 | 7 | 0.87 | 0.22 | 40 ms |
| smoke student | 7.65 st | 3.51 | 29 | 0.37 | 0.66 | 44 ms |

- **Melody transfers fast:** 4.7 → 7.7 st in 10 minutes.
- **Identity and quality are the risk:** trained on a different voice, it stops sounding like amy. This is exactly why the real run trains on audio *converted into amy's voice*, and why early checkpoints are worth evaluating.
- **One criterion adjusted before any real student was scored:** TTFA "within 10% of amy" failed at 44 vs 40 ms. The architecture is identical, so that's jitter. The measurement now uses 7 runs (was 3) and a 5 ms tolerance.

## Full conversion

All **1,023 clips converted, 0 failures** (Seed-VC f0-conditioned, 30 diffusion steps, 44.1 kHz, amy reference 25 s). Four T4 containers took 4,230 / 4,313 / 4,408 / 5,777 s, **5.2 GPU-hours in total (about $3.10)**, about 18 s of GPU per clip on average.
- The run went through `modal run --detach`, which also re-launched cleanly after the first attached attempt was stopped.
- The laptop client reported occasional heartbeat timeouts. Being detached, the job was unaffected.

## Quality gate

*(see the gate report: [gate_report_dailytalk_f1_seedvc_f0.txt](../../experiments/results/gate_report_dailytalk_f1_seedvc_f0.txt))* **757 of 1,023 kept (74%), 39.6 min.**
- Dropped for one reason alone: 159 breakage alarm (UTMOS < 3.5), 63 words, 21 pitch jumps.
- 198 word mismatches were forgiven because the human original had the same count (casual speech vs transcript).
- The kept set is 36% questions and 11% openers.

## Listening check of the training audio, and a change of goal

The converted clips (`61_dt_seedvc_f0.wav`) were heard as an unnatural voice, though the prosody and delivery transferred. Since a student learns whatever its training audio sounds like, training only on converted audio risked teaching that unnatural voice along with the delivery — and keeping amy's identity was no longer required. The goal was therefore changed from "amy with better delivery" to "a proper, natural voice" ([D38](../decisions.md)).

A student learns whatever its training audio sounds like. So training only on converted audio risks teaching the "unnatural voice" along with the delivery. Three runs, the same 2-hour cap each (L4):

| Run | Training audio | Change | Status |
|---|---|---|---|
| A `20260926-0109` | converted (757 clips) | everything trains | **stopped after ~15 min** (would learn the converted voice) |
| B `20260926-0120-freeze-decoder` | converted (757 clips) | **decoder frozen** (the part that renders the waveform, largely the voice) to keep amy's clean sound while the rest learns delivery | **stopped after ~20 min** ([D39](../decisions.md): by design it blends amy's sound with another voice's delivery, violating the one-clean-voice requirement) |
| C `20260926-0121` | **original human recordings** (1,023 clips, 50.8 min) | everything trains; the student becomes the human speaker's voice, as Piper voices normally are | running; extended in stages until it is **one clean voice** (D39) |

### The run C speaker

DailyTalk ([Lee et al., arXiv:2207.01063](https://arxiv.org/abs/2207.01063)):

- **"Two English-fluent speakers were employed as voice actors, each having lived in the US for at least 3 years."**
- Recorded in a studio at KAIST, South Korea. The actors "recorded actual conversations rather than just reading the script", and were asked to add fillers (*uh, um*) in about half the dialogues.
- The female speaker has 11,906 clips, 39,124 s (about 10.9 h) in total; 1,023 clips (50.8 min) from the first data shard were used.

The speaker's accent was judged non-American but acceptable, and her raw voice was preferred from a training clip (`dt00166_09.wav`). A student trained on her recordings learns her accent. The input phonemes stay espeak `en-us`, and the model learns her realisation of them. Some Whisper mismatches in the gate were likely accent, which is why a mismatch was forgiven when the original recording had the same one.

## Training runs

### Run C midpoint (about 1 h, epoch ~6790; [output](../../experiments/results/student_runC_midpoint.txt))

| | Core pitch range | Drop before pause | UTMOS (alarm) | Wrong words / 40 | Similarity to amy | Similarity to her | TTFA | × real time |
|---|---|---|---|---|---|---|---|---|
| amy | 4.77 st | +0.5 dB | 4.42 | 7 | 0.87 | 0.22 | 35 ms | 23.4 |
| **run C, 1 h** | **7.78 st** | **−0.1 dB** | 3.88 | 26 | 0.29 | **0.76** | 31 ms | 24.5 |

Her own recordings' consistency is 0.62; her recordings vs amy score 0.21.

- **One voice, not a blend:** the student already sounds like her (0.76, more consistent than her own varied recordings) and is as far from amy as her real voice is (0.29 vs the 0.36 limit). The identity transfer is complete within an hour.
- **Delivery improved:** a 63% wider melody than amy (7.8 vs 4.8 st; the human has 9.0), and softer endings (−0.1 vs +0.5 dB). The "Well," contour now rises at the end instead of falling.
- **Not finished:**
  - 26 wrong words vs amy's 7;
  - the breakage alarm trips (UTMOS 3.88 vs 4.42).

  VITS keeps removing artifacts through its adversarial losses long after the mel error flattens (piper's own note in `__main__.py`), so more training is the expected fix. Some of the word mismatches may be her accent: Whisper also mismatched her real recordings.
- Speed unchanged (the same architecture), as D32 requires.

**Listening verdict (midpoint clips):** voice breaking on the first clips, unnatural delivery — judged worse than the converted clips. The metrics had correctly flagged trouble (26 wrong words, and the alarm), but the headline numbers (melody, one voice) looked good. **Listening overruled the good-looking numbers, as intended (D37).**
- **Why the converted clip sounds better:** `61_dt_seedvc_f0` is not text-to-speech. It is the human's *recorded performance* (timing, slang, melody) re-voiced by a large diffusion model. The delivery is copied from a person. The student must *generate* delivery from text with ~16M parameters. So 61 is closer to an upper bound than to a fair competitor, and a small TTS student may not reach it.
- "Voice breaking" is typical of a partly trained VITS fine-tune (pitch and timbre not yet stable).

**Decision ([D40](../decisions.md)):**
- Let run C finish its stage.
- **Resume run A** (converted audio, everything trains, so one voice — the 61-style delivery that listening preferred) from its checkpoint at epoch 6699, for 2 h.
- Compare A and C at matched training time, by ear.

## Evaluation

*(pending)*

### Is the shaking measurable? ([E15](../../experiments/e15_voice_stability.py), [output](../../experiments/results/e15_voice_stability.txt))

The converted clips were then also heard as shaky/breaking, so run A (which learns from them) was stopped. Standard Praat voice-stability measures, medians over the same 10 held-out sentences:

| Condition | Jitter | Shimmer | HNR |
|---|---|---|---|
| human original (DailyTalk) | 1.46% | 7.23% | 13.7 dB |
| amy | 1.66% | 6.20% | 13.2 dB |
| kNN-VC | 1.86% | 6.99% | 14.0 dB |
| Seed-VC | 1.63% | 7.10% | 11.2 dB |
| Seed-VC f0 (the training audio for A) | 1.78% | 7.81% | **11.9 dB** |
| run C, 1 h | 1.76% | 6.80% | 13.5 dB |

- **The converted audio is measurably rougher:** the lowest HNR, and more pitch and loudness wobble than the human. That matches the ear.
- **The run C student's "breaking" is barely visible here:** jitter is 20% above the human's, but HNR equals the human's. What is heard is likely pitch breaks or unstable intonation over a phrase, which cycle-level measures miss.
- These measures are defined for sustained vowels; on connected speech every voice exceeds the textbook norms, so only the comparisons are meaningful.

The stated stopping rule: **if run C still breaks at the end of its 2 h stage, stop and report a negative result; amy stays the default.**

## Run C at 1.5 h (2026-09-26 03:00)

Exported mid-stage (tag `-1h30`, all 11 checkpoints verified). Same held-out test as the midpoint:

| | melody (st) | end drop (dB) | UTMOS | wrong words | sim amy | sim her | jitter % | HNR dB |
|---|---|---|---|---|---|---|---|---|
| amy | 4.75 | +0.9 | 4.41 | 7 | 0.86 | 0.22 | 1.66 | 13.2 |
| run C, 1 h (mid) | 7.78 | −0.1 | 3.88 | 26 | 0.29 | 0.76 | 1.58 | 13.5 |
| run C, 1.5 h | 8.23 | −0.8 | 3.98 | 20 | 0.29 | 0.77 | 1.70 | 13.2 |
| her real recordings | 9.00 | | | | 0.21 | | 1.46 | 13.7 |

Improving slowly, 6/9 criteria (fails: melody ≥ 9 st, UTMOS alarm, wrong words ≤ amy). By the stability measures it is no shakier than amy.

**Training data length.** One hypothesis was the short training clips (median 2.6 s). The raw set is 1,023 clips, 50.8 min. Median clip length is 2.6 s (10th–90th percentile 1.3–5.3 s). 78% of clips are under 4 s, 41% contain more than one sentence, and 36% contain a comma. However:
- amy's own training set is about the same (1,019 clips in ~1 h, so ~3.5 s on average), so clip length alone doesn't explain the gap to amy.
- Piper synthesizes one sentence at a time, so multi-sentence training clips would add little at inference.

The bigger difference is **quantity**: 51 of her ~10.9 h were used, chosen originally as a conversion test set. The next run should use all of her speech except the held-out conversations, including her longest turns.

**Garbled words in the middle of the paragraph.** Each piece of the reference paragraph was checked with local Whisper. amy made 0 errors. The 1.5 h student garbled these, identically at 1.0× and 1.2×:
- "it's about cadence, rhythm," heard as "It's of our cadence. Welcome!";
- "Does it handle subtle micro-breaks, like after a semicolon;" heard as "since it handles subtle micro-breaks like afters and colon" (5 wrong);
- "dead" heard as "that";
- "can" heard as "can't".

The garbled words are rare, bookish words ("rhythm", "semicolon", "micro-breaks") absent from everyday conversation. Fine-tuning on 51 min of dialogue makes the model forget how amy pronounced them. More, and more varied, data is the fix. Candidate: Expresso (Meta, 2023): 40 h studio 48 kHz, 4 speakers (2 female), 11 h read (transcribed) + 30 h improvised dialogue (not transcribed), 26 styles including emphasis and emotions. **License CC BY-NC 4.0 (non-commercial).**
