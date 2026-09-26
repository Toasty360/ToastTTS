# ToastTTS: handoff

*Written 2026-09-26 03:20. Read this first, then [research/README.md](research/README.md).*

## What ToastTTS is

An on-device (laptop CPU) streaming TTS layer that makes a small Piper voice sound natural:
- it pauses like people do;
- it softens the ends of phrases before a pause instead of cutting to silence;
- it starts fast;
- it speaks an LLM's output while the LLM is still writing.

The project is organised as research: [notebook](research/notebook/), [decisions D1–D41](research/decisions.md), [methodology](research/methodology.md), [benchmarks](benchmarks/README.md).

**About the author:**
- New to TTS jargon, so explain in plain words.
- Judges by ear, and **listening overrides metrics**.
- Often away from the keyboard: send audio with SendUserFile.

## State right now

| Area | Status |
|---|---|
| Engine (`toast/`) | **Done and working.** Default voice amy-medium at 1.2×. |
| Dynamic-input entry point | Done: `toast/engine.py` (`ToastEngine`), `scripts/say.py`, `scripts/live.py` |
| Word emphasis | **Prototype works**, not yet in the streaming engine (see below) |
| Emotions | Options given to the author; **no choice yet** |
| Training a better voice | **Paused.** Run C stopped at 03:15; next dataset undecided (see below) |
| Modal | **Nothing running.** All `toasttts-*` apps stopped. |

**Engine benchmark** (amy 1.2×, laptop CPU; [`experiments/results/engine_benchmark_amy_1.2x.txt`](experiments/results/engine_benchmark_amy_1.2x.txt)):

| Measure | Value |
|---|---|
| Load time | 1.6 s |
| Process memory | 166 MB |
| TTFA, full text | 31 ms (28.6× real time) |
| TTFA, streaming at 30 tok/s | 99 ms, 0 stalls |
| TTFA, streaming at 10 tok/s | 229 ms, 0 stalls |
| TTFA, short sentences | 33 ms median |

## Rules the author set (don't break these)

- **No OpenAI Whisper API** (paid `whisper-1`). Use local Whisper (`toast/word_check.py`, small.en).
- **Soniox / `cloud_voices/` outputs must not be used for training or tuning** (Soniox terms, D35). They are listening references only.
- **Kokoro is a reference only.** Don't adopt its voices, and don't try to "fix" Kokoro.
- **No heavy model.** Any trained voice must be the same size and speed as Piper-medium (D32): TTFA and speed within 10% of amy.
- **UTMOS is not trusted** as a naturalness score. It is only a breakage alarm. A blind A/B test (`distill/blind_test.py`) decides (D37).
- **One clean voice.** No blended or "two voices overlapped" results. It doesn't have to be amy.
- **Use GPU efficiently** (cost is fine, waste isn't), and **never lose a run's output**:
  - checkpoints commit to the Volume every 5 min;
  - export is separate and re-runnable;
  - a single stage is capped at ≤ 7.5 h.

## The training thread (where we stopped)

Goal: a Piper-medium voice with natural conversational prosody. Full story in [notebook 15](research/notebook/15-training.md) and the [distillation plan](research/distillation-plan.md).

| Run | Data | Outcome |
|---|---|---|
| A, B | DailyTalk female converted to amy's voice with Seed-VC | Stopped. The converted audio sounds shaky or breaking (E15: HNR 11.9 dB). |
| C `20260926-0121` | DailyTalk female (speaker 1), **raw**, 1,023 clips, 50.8 min, fine-tuned from amy | Stopped at ~1.9 h. Best export at 1.5 h (below). |

**Run C at 1.5 h** (`models/en_US-amy_distill_20260926-0121-1h30_*`; audio `out/listen/80_20260926-0121-1h30_*`):

| Voice | Melody (st) | Wrong words | UTMOS |
|---|---|---|---|
| amy | 4.75 | 7 | 4.41 |
| Run C, 1.5 h | 8.23 | 20 | 3.98 |
| Her real recordings | 9.0 | – | – |

- It is clearly her voice, not blended (similarity to her 0.77, to amy 0.29).
- It is no shakier than amy by jitter and HNR.
- It passes 6/9 criteria.
- **The author heard gibberish in the middle of the paragraph.** Local Whisper located it: "it's about cadence, rhythm," became "It's of our cadence. Welcome!", and "like after a semicolon" became "like afters and colon". Rare words get garbled.

**Diagnosis:** too little, too narrow data. We used 51 of her ~10.9 h. That set was originally chosen as a voice-conversion test set, not for training. The author also suspected the short clips (median 2.6 s). But amy's own training clips average ~3.5 s, and Piper speaks one sentence at a time, so clip length is a minor factor.

**Pending decision.** Ask the author: *is ToastTTS for commercial use?*
- **Non-commercial (research, paper, personal):** use **Expresso** (Meta 2023, CC BY-NC 4.0).
  - 40 h of studio audio at 48 kHz from 4 American speakers (2 female).
  - 11 h of read speech with transcripts, plus 30 h of improvised dialogue without transcripts (one channel per actor), to be transcribed with local Whisper.
  - 26 styles, including emphasis and emotions. These could become switchable "speaker" IDs, which would give emotions.
  - Pick one female speaker.
- **Commercial:** use all of the DailyTalk female's ~10.9 h (CC BY-SA 4.0), held-out conversations excluded, plus some read speech so rare words survive.

The next run needs a new dataset builder (like `distill/make_raw_dataset.py`), then `distill/train_modal.py --dataset <name>`. It will need longer than 2 h: use `--resume` stages.

## Word emphasis ([notebook 16](research/notebook/16-word-emphasis.md), D41)

The LLM marks the focus word: `I didn't say *he* stole it.` `toast/emphasis.py` then performs it with Praat PSOLA on Piper's phoneme timings:
- a pitch accent over the word;
- lengthening;
- extra loudness;
- post-focus compression.

Piper's timings require an alignment-patched model: `models/aligned/en_US-amy-medium.onnx`.

- The **"subtle"** preset was barely noticed. The **"strong"** preset (≈ +6 st) was *"noticed clearly"*. Both kept 0 wrong words.
- **Next:** integrate it into the streaming engine. Markers must survive `StreamChunker`, the aligned model must be used, and an LLM prompt must mark contrastive focus. Default to "strong".

## Emotions (awaiting the author's choice)

1. **Engine presets** (recommended first). LLM tags like `[happy]` map to pitch level and range, rate and energy per sentence, reusing the emphasis machinery. It is cheap and keeps the voice.
2. **Emotion-conditioned student.** Train style or emotion IDs: from Expresso styles if non-commercial, otherwise from DailyTalk's emotion labels.
3. **Big expressive models.** Too slow on CPU, so it breaks the D32 hard constraint.

## Commands

The venv is Python 3.12 (`uv`). Run everything from the project root.

```powershell
.venv\Scripts\python scripts\say.py "Hello there."                  # speak any text (engine)
.venv\Scripts\python scripts\bench_engine.py                          # CPU benchmark
.venv\Scripts\python scripts\emphasis_demo.py --strong                # emphasis demo
.venv\Scripts\python distill\evaluate_student.py piper:<voice>        # score a trained voice vs amy
.venv\Scripts\python experiments\e15_voice_stability.py 20260925-2232-dailytalk piper:<voice>
.venv\Scripts\python distill\blind_test.py piper:<voice>              # blind A/B for the author
```

**Modal** (profile `toasty360`; volumes `toasttts-cache` and `toasttts-train`):
- Launch with `modal run --detach`, so the laptop can sleep.
- Stopping needs `--yes`.

```powershell
.venv\Scripts\python -m modal run --detach distill\train_modal.py --dataset <name> --hours 2
.venv\Scripts\python -m modal run distill\train_modal.py --export-only <run_id> --tag -<label>
.venv\Scripts\python -m modal run --detach distill\train_modal.py --resume <run_id> --hours 2
.venv\Scripts\python -m modal app list ; .venv\Scripts\python -m modal app stop <app_id> --yes
```

## Known quirks

- **PowerShell writes a BOM:** read text with `utf-8-sig`. `say.py` strips it from stdin.
- **`python -c` quoting breaks in PowerShell:** put scratch scripts in files.
- **External monitor speakers swallow the first words:** laptop output is fine. `LivePlayer` idle room tone plus a 0.6 s wake helps. Shelved at the author's request.
- **The word checker normalises before comparing:** times, numbers, British spellings, compounds and contractions (`toast/word_check.py`). It isn't raw WER.
