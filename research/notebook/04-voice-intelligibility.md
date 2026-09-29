# 04: Intelligibility, and why the voices were changed

*2026-09-25 · Experiment [E05](../../experiments/e05_word_clarity_by_voice.py) · Scan: [`scripts/find_voices.py`](../../scripts/find_voices.py), data [`benchmarks/voice_scan_en_US-libritts_r-medium.csv`](../../benchmarks/voice_scan_en_US-libritts_r-medium.csv) · Code: `toast/word_check.py`, `toast/pronounce.py`*

## The problem

After two rounds of fixes (see [02](02-natural-pauses.md)), listening tests revealed that "semicolon" was never spoken in the audio. Measuring audio length could only show the word was compressed: the phrase with "semicolon" was about 0.25 s longer than without it, while the word alone takes about 0.5 s. Determining what was actually said required transcription; duration analysis could not answer it.

## New instrument: the word check

Whisper small.en (local) transcribes the generated audio, and the transcript is aligned with the input text to count wrong words. This became a permanent project tool: `render.py --check-words`, `bench_voices.py`, `find_voices.py`. Details in the [methodology](../methodology.md).

## Findings

**1. It's the voice model, not the pipeline.** In E05 the plain model was called directly (no splitting, trimming or fades), 2 takes per sentence ([output](../../experiments/results/e05_word_clarity_by_voice.txt)):

| Voice | "A semicolon is useful here." heard as | "It is about cadence." |
|---|---|---|
| libritts_r-medium 3922, file defaults | "A sema…", "A sema…" | "is about kids", "…kittens" |
| libritts_r-medium 3922 @0.5/0.8 | "a cement…", "A semit…" | correct once, "kittens" once |
| libritts-high p3922 @0.5/0.8 | "A sema…", "cement…" | "Cairns" once |
| **lessac-medium** | correct ×2 | correct ×2 |
| **amy-medium** | correct ×2 | correct ×2 |

lessac and amy got all 4 test sentences right in both takes. Both libritts models failed on "semicolon" every time.

**2. It's the whole model, not one speaker.** `find_voices.py` scored 60 of the 904 libritts_r speakers, evenly spread, on three tricky sentences. **None** said "semicolon" correctly: sema / sima / semi / semis / summer. Many also said "subtle" as "civil" or "several". The best speakers still had 4% word error, all from that word.

**3. The high-quality libritts model is no better on this, and 5× slower:** about 100 ms for a short piece and 7× real time, versus about 20 ms and 33× for libritts_r-medium.

**4. Respellings don't help** (E05, 3 takes each):

| Spelling given | Pronunciation Piper uses | Heard |
|---|---|---|
| semicolon | ɐ sˌɛmɪkˈoʊlən | summon, cement, cement |
| semi-colon | ɐ sˈɛmaɪkˈoʊlən | semi, semay, semi |
| semi colon | ɐ sˈɛmaɪ kˈoʊlən | semive, semi, semi |
| semmy-colon | ɐ sˈɛmikˈoʊlən | semi, semi, semi |

The pronunciation is correct; the voice still drops "-colon".

**5. A fix that breaks something else.** Piper accepts exact pronunciations inside `[[ ]]`, but the text around the brackets is then processed on its own, and "a" before the bracket turned into the letter "AY". Respellings keep neighbouring words in context, so `toast/pronounce.py` uses respellings (and says why).

## Likely explanation

libritts / libritts_r are trained on hundreds of audiobook speakers with little data each. lessac and amy are single-speaker models trained on much more data from one person. Uncommon words suffer in the multi-speaker models.

## Decision

Switch the default voice from libritts_r 3922 to **lessac-medium**: 0 wrong words out of 95 in two paragraph takes, 38 ms TTFA, 22× real time. Listening tests confirmed the voice sounded good and the semicolon issue was fixed. Later replaced by amy after the full benchmark ([05](05-voice-benchmark.md), [07](07-cloud-reference-and-pacing.md)).

Paragraph check at the time (`render.py --check-words`, one take):

| Recording | TTFA | × real time | Wrong words / 95 |
|---|---|---|---|
| Kitten mini, sentence split | 4440 ms | 1.9× | 9 (mostly contractions, later fixed in the checker) |
| 3922 @0.5/0.8 | 30 ms | 27× | 1 ("semicolon" → "sentence") |
| **lessac-medium** | **38 ms** | **22×** | **0** |
| lessac-high | 156 ms | 4.9× | 0 |
| libritts-high p3922 | 129 ms | 5.2× | 8 |

## Lesson

**Measure what was said, not just how long it took.** Duration and energy analysis kept producing plausible but wrong explanations (see the corrections in [02](02-natural-pauses.md)). Transcription settled the question in minutes.

→ Next: [05: benchmarking every voice](05-voice-benchmark.md)
