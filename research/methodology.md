# Methodology

How every number in this project is measured. Each metric lists the code that computes it, so any result can be traced and re-run.

## Test environment

All measurements were taken on one laptop unless stated otherwise.

| | |
|---|---|
| CPU | Intel Core i7-13700H (14 cores / 20 threads), CPU only, no GPU used |
| RAM | 16 GB |
| OS | Windows 11 |
| Power mode | "Balanced" until the Coval run on 2026-09-25, then "Best performance". Timings from before/after the switch are not directly comparable (see [notebook/06](notebook/06-coval-comparison.md)) |
| Python | 3.12.14 (KittenTTS needs `misaki`, which does not support 3.13+) |
| Key packages | piper-tts 1.8.0, onnxruntime 1.30.0, kittentts 0.8.1, faster-whisper 1.2.1, torch 2.14.0, whisper-normalizer 0.1.15, jiwer 4.0.0 |

Laptop timings drift by roughly 20–30 ms between runs (heat, background load). Scripts therefore repeat measurements and report medians, and comparisons between voices are always measured in the same session.

## Test texts

| File | What it is | Used for |
|---|---|---|
| [`samples/reference.txt`](../samples/reference.txt) | 95-word paragraph written to stress pacing: asides, lists, a time, a semicolon, an ellipsis, questions | Listening tests, pacing, word checks |
| [`samples/word_test.txt`](../samples/word_test.txt) | 51 words of varied vocabulary (committee, cathedral, colleague's…) | Word checks, so voices aren't judged on one paragraph |
| [`samples/no_early_comma.txt`](../samples/no_early_comma.txt) | Sentence with no early comma, "3 PM", "Dr. Smith" | Streaming edge cases |
| [`samples/coval_tts_v1.json`](../samples/coval_tts_v1.json) | Coval's 30 customer-service prompts (Apache-2.0) | Comparison with cloud providers |

## Metrics

### TTFA (time to first audio)

The time between "here is the text" and the first sound that can be played. The model is loaded and warmed up first, as in a running app. There are three variants, and every result states which one it uses:

| Variant | Starts when | Ends when | Code |
|---|---|---|---|
| **Full text** | The whole text is handed over | The first audio chunk is ready | `toast/metrics.py` (`scripts/render.py`) |
| **Live** | The first LLM token arrives (fake LLM, fixed rate) | The first audio chunk is ready | `scripts/live.py` |
| **Coval-style** | The whole text is handed over | The first chunk is ready, **plus** any silence before the first audible sample (a 10 ms frame with RMS > 0.01, stepped every 1 ms: Coval's rule) | `scripts/bench_coval.py` |

Cloud TTFA (from Coval) includes a network round trip. Ours runs on the device and has none. The two are shown side by side but are **not a like-for-like race**.

Reported as the median of several runs (3 for `render.py`, 60 clips for Coval-style).

### Stalls

Playback is simulated in real time while audio is being generated. A stall is any moment where the queued audio runs out before the next chunk is ready, which would be heard as an unplanned gap. The target is always **0**. Code: `toast/metrics.py`; real playback: `toast/player.py`.

### Speed (× real time)

Seconds of audio produced per second of compute. Anything above 1× can keep up; higher means more headroom.

### Wrong words (our word check)

Whisper **small.en** (int8, local) transcribes the audio, and the transcript is aligned against the input text word by word (substitutions + deletions + insertions). Code: `toast/word_check.py`.

Before comparing, both sides are normalized so that *writing style* isn't counted as a *speaking* mistake. Each rule was added after it caused false errors (see [notebook/05](notebook/05-voice-benchmark.md)):

| Whisper writes | We wrote | Treated as |
|---|---|---|
| `3.15 p.m.` / `3:15pm` | `3:15 PM` | same |
| `one hundred and eight` | `108` | same (numbers → words, no "and") |
| `harbour`, `neighbourhood` | `harbor`, `neighborhood` | same (small British→American list) |
| `microbreaks`, `ultralowlatency` | `micro-breaks`, `ultra-low latency` | same (joined/split compounds) |
| `is not`, `it is` | `isn't`, `it's` | same (contractions expanded) |

Tests: `tests/test_word_check.py`. Known limits: Whisper can mishear clear speech, and one voice (`en_GB-cori-medium`) had a transcript that dropped about 30 words for unknown reasons. A difference of 1–2 words between voices is noise.

### WER, Coval method

A replication of [Coval's open-source TTS benchmark](https://github.com/coval-ai/benchmarks), used only for the cloud comparison (`scripts/bench_coval.py`):

- Transcription by OpenAI **`whisper-1`** (the same model Coval calls), or locally by Whisper **large-v2** (`--asr local`; `whisper-1` is large-v2).
- Both texts normalized with `whisper_normalizer`'s `EnglishTextNormalizer`; edit distance via `jiwer`.
- **Pooled** over all clips: total errors ÷ total reference words (Coval's headline aggregation).
- 30 prompts × 2 takes per voice. Coval samples 10 prompts per run, about 480 clips a day.

### Naturalness (predicted MOS, 1–5)

**UTMOS22 strong** (via `tarepan/SpeechMOS` v1.2.0), a model trained to predict human naturalness ratings. Code: `toast/naturalness.py`.

- In the voice benchmark it is the mean over the 5 sentences of `reference.txt`, each rendered through our pipeline.
- For whole recordings it is scored on the first 20 s.
- The two methods give different absolute values for the same voice (lessac-medium: 3.82 on a 20 s clip, 4.32 per sentence). Only compare numbers produced by the same method.
- On a single 20 s clip, differences under about 0.15 are noise. It is a predictor, not a listening test: listening decisions were made by ear.
- **It disagrees with this listener in ways that matter:** ryan-medium scored 4.41 (near the top) but was heard as unnatural; Kokoro and amy scored the same (4.37 vs 4.38) but were heard as clearly different; it couldn't separate softening levels. From D37 on, UTMOS is used only as a **breakage alarm** (it does catch badly damaged audio, e.g. a converted clip at 2.85), never as evidence that something sounds *more natural*. That judgment is made by a **blind A/B listening test** (`distill/blind_test.py`).

### Speech pacing

Whisper word timestamps (`toast/speech_stats.py`):

- **Words per minute:** overall, and while talking (pauses of 80 ms or more removed).
- **Pause lengths:** grouped by the punctuation before the pause.

Word boundaries from Whisper are approximate (tens of ms), and the gaps include the quiet edges of words. So these numbers are only compared against each other, never against our configured pause table.

## Listening tests

The main quality judgments were made by one listener (the project author) with headphones, comparing numbered files in `out/listen/`. Automatic metrics were added to make those judgments checkable and to catch what a listener can miss. They never replaced listening: for example, ryan-medium had the best word error rate but was rejected by ear for sounding less natural.
