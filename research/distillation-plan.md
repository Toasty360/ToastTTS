# Distillation plan: Kokoro-like delivery in Piper's own voices

*Drafted 2026-09-25, revised the same day · Status: **superseded** · Budget: Modal credits (~$20)*

> This plan is kept as a record of the original design. The Kokoro-teacher approach was replaced by openly licensed human speech ([notebook 14](notebook/14-human-source-and-converter.md)), and the voice-conversion + fine-tuning method was attempted in [notebook 15](notebook/15-training.md).

**Revision note:** the first draft trained the student directly on Kokoro's audio, which would have produced a Kokoro-sounding voice. Revised: Kokoro provides only the **way of speaking** (melody, timing, soft endings). Each student keeps **its own Piper voice**, and several voices and accents are the goal.

## Goal

Fast Piper voices, starting with amy, that keep their own sound but speak with Kokoro-like delivery:
- a wider, livelier melody;
- a "Well," that rises before it falls;
- softer endings before pauses.

They must stay at Piper speed (about 0.1 s to first audio, about 25× real time) and come in several voices and accents.

## Hard constraint: the student is not a heavier model

- **What ships is a Piper medium model, identical in architecture and size to amy:** ~15.8M parameters, 63 MB (Kokoro: ~81.4M, 326 MB). Fine-tuning changes the *values* of amy's weights, not their number, so inference cost is unchanged by construction.
- **Kokoro and the voice-conversion model run only offline**, once, to create training audio. They never ship and never run inside ToastTTS.
- **The acceptance check enforces it:** a student whose TTFA or real-time speed is more than 10% worse than amy's (34 ms, 25× in `bench_voices.py`) fails, however good it sounds. A Piper-high student (~4× real time) is **not** an acceptable fallback under this constraint; it's listed under risks only as a last resort requiring an explicit decision.

## Why

- **The gap is delivery, and it's measured** ([12](notebook/12-robotic-melody-and-well.md)): amy's pitch range is 6.8 semitones vs 10.5 for Kokoro af_heart and 13.0 for Soniox; amy's "Well," falls like a full stop.
- **Post-processing only partly closes it** (E11): stress and phrasing come from what a model learned.

## Approach: Kokoro speaks, the audio is re-voiced, Piper learns

```
text ──► Kokoro (teacher: melody + timing) ──► voice conversion ──► "amy's voice, Kokoro's delivery" ──► fine-tune amy
                                                   ▲
                               reference: lots of amy's own audio (free to generate)
```

1. **Kokoro** speaks the training sentences, giving natural melody, timing and soft endings.
2. A **voice-conversion (VC)** model re-voices that audio as the target Piper voice. It replaces *who* is speaking and keeps *how* it's said. The target's reference audio is unlimited, because it can be generated with Piper.
3. The target voice's own Piper checkpoint is **fine-tuned** on the converted audio, so it learns the new delivery while starting from its own voice.

**Accent:** VC keeps the source's pronunciation, so each target is paired with a Kokoro teacher of the same accent (British Piper voices learn from a British Kokoro voice).

## Step 0: feasibility test (laptop, free, ~1 h). Nothing else starts until this passes.

The whole plan depends on one unproven link: **does voice conversion keep Kokoro's melody?** Some VC methods rebuild speech from the *target's* recorded frames, which can pull the melody back toward amy's flat one.

- Convert 10 Kokoro `af_heart` sentences into amy's voice with two open VC models:
  - **kNN-VC** (`bshall/knn-vc`, MIT license): simple, any-to-any, runs on CPU.
  - **OpenVoice v2 tone-color converter** (MIT license): designed to change timbre while keeping the source's delivery.
- Measure each result:

| Check | Tool | Pass if |
|---|---|---|
| Melody kept | E11 pitch range | ≥ 9 st (Kokoro 10.5, amy 6.8) |
| "Well," kept | E12 contour | rises before it falls |
| Sounds like amy | listening, plus a speaker-similarity score | recognisably amy |
| Words intact | word check | 0 wrong words |
| Artifacts | listening | no metallic or garbled sound the student would learn |

- **Pass:** continue with the better VC model. **Fail:** stop here and write it up. Fallbacks would be pitch-contour transfer by time alignment, or accepting Kokoro-voiced students; both are weaker.

## Step 0 result (2026-09-25): passed with kNN-VC, one open problem

See [notebook/13](notebook/13-vc-feasibility.md).
- **kNN-VC keeps 91% of Kokoro's core melody, sounds like amy (0.75 vs amy's own 0.84), and has 0 wrong words and no pitch glitches.** OpenVoice v2 fails.
- **Open problem: kNN-VC outputs 16 kHz** (energy above 7 kHz is 0.21% vs amy's 1.05%), so the student could learn a muffled top end.

**Step 0b (before any training spend):** decide how to keep full bandwidth. Options: mix in some of amy's own full-band audio; a 22/44 kHz converter such as Seed-VC (check its license); or bandwidth extension. Test on a few clips first, the same way as Step 0.

## Never lose a GPU run (applies to every Modal job)

A failed final export once lost a whole training run, hence these safeguards:

- **Smoke first:** every job has a `--smoke` mode that runs the whole chain (including saving and fetching) on a tiny input.
- **Persist as you go:** results and **training checkpoints go to a Modal Volume and are committed as soon as they exist**, before any later step can fail.
- **Export after saving, never instead of saving:** the raw PyTorch checkpoint is saved and committed first. ONNX export runs as a separate step (in `try`, verified with onnxruntime in the container), so a failed export never costs the training.
- **Fetch without recompute:** a `--fetch` mode pulls finished results from the Volume.
- **Laptop saves before it evaluates**, so a bug in evaluation never means re-running the GPU.
- **Hard timeouts, cheapest adequate GPU, models cached in the Volume.**

## Voices

Candidates with public training checkpoints (`rhasspy/piper-checkpoints`) that did well in the voice benchmark ([05](notebook/05-voice-benchmark.md)):

| Student (keeps its voice) | Accent / gender | Our benchmark | Kokoro teacher (same accent) |
|---|---|---|---|
| **amy-medium** | US, female | 4.38, 0 wrong words | `af_heart` |
| bryce-medium | US, male | 4.22, 0 | `am_puck` |
| jenny_dioco-medium | UK, female | 4.21, 0 | `bf_emma` |
| alan-medium | UK, male | 4.05, 2 | a `bm_*` voice (choose by listening) |

**Phases:**
- **Phase 1:** amy alone. This proves the method with the least spend.
- **Phase 2:** the other voices, either as separate fine-tunes or as **one multi-speaker model** (Piper supports speaker-labelled data). The older Piper trainer could convert a single-speaker checkpoint to multi-speaker (`--resume_from_single_speaker_checkpoint`); whether piper1-gpl still can must be checked. Phase 2 may need more than the remaining credits; it will be costed from Phase 1's measured numbers.

## Steps (Phase 1: amy)

1. **Text** (laptop, free). About 1,000 sentences, the same size as amy's own original fine-tune (1,019 sentences, about 1 h).
   - Source: OpenAssistant oasst1/oasst2 assistant replies, split into sentences (Apache-2.0). They're conversational: questions, openers like "Well,", numbers.
   - Filter to 4–30 words, with no code or URLs.
   - Held-out test texts are removed: `reference.txt`, `word_test.txt`, the Coval prompts.
   - Our `text_normalize` rules are applied.
2. **Teacher audio** (laptop, free, ~20 min): Kokoro `af_heart` speaks each sentence whole.
3. **Voice conversion** (laptop, or GPU if too slow): convert to amy using about 10 minutes of amy's own Piper audio as reference.
   - **Quality gate:** drop any clip with a Whisper word error or pitch range under 8 st, and resample to 22.05 kHz.
4. **Pilot training** (Modal L4 or A10G, ≤ 0.5 GPU h, ≤ $0.55): fine-tune from `en/en_US/amy/medium/epoch=6679-step=1554200.ckpt` with piper1-gpl; measure steps per second, memory and cost per hour; check that the ONNX export loads in `toast/voices.py`.
5. **Main training** (≤ 8 GPU h, ≈ $6.50–9): checkpoint about every 30 minutes, export and score each one automatically (step 6 metrics), stop early on a plateau, with a hard job timeout.
6. **Evaluate** on held-out text only:

| Measure | Tool | amy now | Kokoro af_heart | **Success if the student…** |
|---|---|---|---|---|
| Still sounds like amy | listening + speaker similarity | — | — | **recognisably amy** |
| Pitch range | E11 | 6.8 st | 10.5 st | ≥ 9.0 st |
| "Well," contour | E12 | fall | rise-fall | rises before it falls |
| Soft last word | E09 | +1.1 dB | −0.7 dB | ≤ 0 dB |
| Artifact alarm (not a naturalness judgment, D37) | UTMOS | 4.38 | 4.37 | no more than 0.3 below amy |
| Wrong words | word check / Coval local WER | 0 / 6.1% | 0 | no worse than amy |
| TTFA and speed | `bench_voices.py`, `live.py` | 34 ms, 25× | 284 ms, 4× | within 10% of amy |
| **Blind listening (decides, D37)** | `distill/blind_test.py`: randomised A/B, hidden key, sign test | | | **preferred over amy** |

7. **Write-up:** notebook entry 13 with every checkpoint's scores, including failures.

## Budget

| Stage | Where | Cost |
|---|---|---|
| Step 0 feasibility, text, teacher audio, conversion, evaluation | laptop | free |
| Pilot | Modal | ≤ $0.55 |
| Phase 1 main run (amy) | Modal, ≤ 8 GPU h | ≤ $6.50 (L4) / $8.80 (A10G) |
| Retry contingency | Modal | ≤ $5 |
| **Phase 1 total** | | **≤ ~$15 of ~$20** |
| Phase 2 (more voices) | Modal | costed after Phase 1; may need more credits |

Rates are Modal's published per-second prices (L4 ~$0.80/h, A10G ~$1.10/h; [modal.com/pricing](https://modal.com/pricing)).

## Risks

| Risk | Mitigation |
|---|---|
| VC flattens the melody | Step 0 measures it before any spend |
| The student learns VC artifacts | Quality gate; listening in Step 0; OpenVoice as the alternative |
| The student learns Kokoro's mistakes (e.g. its pause inside times) | Word-check gate; times written as words |
| Overfitting | Held-out evaluation; early stopping |
| Too little capacity | Measured; try 2,000 sentences. A Piper-high student would break the speed constraint, so it needs an explicit decision and isn't a default fallback |
| Cost overrun | Pilot first, hard timeouts, per-second billing |
| Licensing | Kokoro, OpenAssistant, kNN-VC and OpenVoice are permissive (Apache-2.0 / MIT). Each Piper voice's own dataset license must be checked before *publishing* a student (amy: "see mimic3-voices"); private use is fine. piper1-gpl is GPL-3.0. `cloud_voices/` is never used ([D18](decisions.md)) |
