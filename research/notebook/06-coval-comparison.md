# 06: How do we compare with cloud TTS providers?

*2026-09-25 · Script: [`scripts/bench_coval.py`](../../scripts/bench_coval.py) · Data: [`benchmarks/coval_tts_v1_whisper1.csv`](../../benchmarks/coval_tts_v1_whisper1.csv), [`benchmarks/coval_tts_v1.csv`](../../benchmarks/coval_tts_v1.csv) · Analysis: [E07](../../experiments/e07_coval_time_format.py)*

## Question

A published comparison ([Gradium, "TTS WER Benchmark 2026"](https://gradium.ai/content/tts-wer-benchmark-2026)) cites the **Coval** leaderboard (benchmarks.coval.ai/tts), which ranks 26 cloud TTS services. Can we run our best voices through the same test and place them on that board?

## Replicating Coval's method

Coval publishes its benchmark code ([github.com/coval-ai/benchmarks](https://github.com/coval-ai/benchmarks)). From the code, not just the docs:

| Part | Coval | Ours |
|---|---|---|
| Prompts | `tts-v1`: 30 customer-service sentences with order numbers, tracking codes, prices, times, names | Same file, copied to `samples/coval_tts_v1.json` (Apache-2.0) |
| Transcription | OpenAI hosted **`whisper-1`** | **`whisper-1`** via the author's API key (about $0.17 of audio at $0.006/min); also Whisper large-v2 locally for comparison |
| Normalization | `whisper_normalizer` `EnglishTextNormalizer` | same |
| WER | pooled: sum of S+D+I ÷ sum of reference words | same (`jiwer`) |
| TTFA | first chunk + leading silence (10 ms frames, RMS > 0.01, 1 ms hop) | same rule, applied to our first chunk |
| Samples | 10 random prompts per run, about 480 clips/day | all 30 prompts × 2 takes = 60 clips per voice |

Our voices ran through the full pipeline (`smart` split, speed 0.9). TTFA was re-measured for all voices together in one session ("Best performance" power mode), because lessac-high had been timed earlier in balanced mode (709 ms at the time, versus 333 ms re-timed).

Cloud numbers are the Coval board for 2026-09-08 as reprinted in the Gradium article, which lists 14 of the 26 models. The live board loads its data with JavaScript, so we couldn't read it directly.

## Results

| Model | WER | P50 TTFA | Runs on |
|---|---|---|---|
| Soniox TTS Rt v2 | 4.0% | 255 ms | cloud |
| ElevenLabs Eleven v3 Conversational | 4.3% | 320 ms | cloud |
| Inworld TTS 2 | 4.5% | 170 ms | cloud |
| Fish Audio S2.1 Pro | 4.7% | 293 ms | cloud |
| Gradium TTS | 4.9% | 214 ms | cloud |
| OpenAI GPT-4o mini TTS | 4.9% | n/a | cloud |
| Deepgram Aura-2 | 5.0% | 290 ms | cloud |
| Rime Mist v3 | 5.0% | 256 ms | cloud |
| **ToastTTS + ryan-medium** | **5.1%** | **75 ms** | this laptop |
| Fluxions vui | 5.3% | 51 ms | cloud |
| Inworld TTS Flash 2 | 5.3% | 75 ms | cloud |
| Cartesia Sonic 3.6 | 5.3% | 440 ms | cloud |
| Palabra TTS v1 | 5.7% | 103 ms | cloud |
| Cartesia Sonic 3.5 | 5.8% | 269 ms | cloud |
| **ToastTTS + amy-medium** | **5.9%** | **97 ms** | this laptop |
| **ToastTTS + lessac-medium** | **6.0%** | **81 ms** | this laptop |
| **ToastTTS + lessac-high** | **6.3%** | **333 ms** | this laptop |
| ElevenLabs Flash v2.5 | 6.5% | 185 ms | cloud |

The local large-v2 scoring agreed within about 1 point: ryan 5.6%, amy 6.1%, lessac 5.4%, lessac-high 5.5%.

**How to read this:**

- **WER is comparable:** same prompts, same transcriber, same scoring code.
- **TTFA is not a like-for-like race.** Cloud TTFA includes the network round trip; ours has none because it runs on the device. That is the point of an on-device engine, but it must be stated wherever these numbers appear.
- **The cloud numbers are one day's snapshot** and move day to day.

## Where our errors come from ([E07](../../experiments/e07_coval_time_format.py))

**About 3.5 WER points of each voice come from how Whisper writes times.** Whisper writes "2:30 PM" as "2.30 pm". The normalizer turns "2:30" into "2 30" but leaves "2.30" alone, so each time costs 2 word errors, and there are 8 times per take. Removing only that formatting difference:

| Voice | WER (whisper-1) | Without time-format errors |
|---|---|---|
| ryan | 5.1% | 1.4% |
| amy | 5.9% | 2.3% |
| lessac | 6.0% | 2.9% |
| lessac-high | 6.3% | 3.0% |

The adjusted numbers are **not** comparable with the leaderboard, because we don't know whether cloud voices were affected. Whisper does write "3.15 pm" for Deepgram and Soniox recordings too ([E06](../../experiments/e06_cloud_pacing.py)), so they probably were. Only the raw WER belongs in a comparison.

**Real mistakes that remain** (across voices; E07 lists amy's in full):

- **Symbols and codes:** "#ORD-24589" is read as "hash O-R-D…" (the pronunciation step spells the symbol out). "E-1047" blurs into "ericode". amy says "$149" in a way Whisper writes as "one.149", and "PTO" is heard as "ptho".
- **Names:** Garcia → "Farcia"/"Parcia", Martinez → "martine is" (both amy takes), David → "Davey", Chen → "Chan".
- **Word slips** (amy, in both takes): "unplug" → "and plug", "zip" → "cip", "logged" → "lobbed". Because these repeat, they're probably pronunciation problems rather than transcription noise.
- Pure formatting that still counts: "login" → "log in".

Both groups are fixable in front of the model: a text-normalization step for symbols and codes, and the pronunciation dictionary (`toast/pronounce.py`) for names.

→ Next: [07: cloud reference recordings and pacing](07-cloud-reference-and-pacing.md)
