# 08: Open questions and where to go next

*2026-09-25*

## The remaining gap is prosody, not pacing

Listening next to Soniox Grace showed the gap clearly: a natural voice knows where to stress a word, how to shape the word before a natural pause, and how to end a question. Our engine now matches cloud voices on speaking rate and pause lengths ([07](07-cloud-reference-and-pacing.md)), and its predicted naturalness is in the same range. But three things are missing:

1. **Stress:** which word in a sentence carries the emphasis.
2. **Pre-pause shaping:** the lengthening and pitch movement on the word *before* a pause.
3. **Question intonation:** a rise at the end of yes/no questions.

**Why Piper can't do these:** it is a ~15M-parameter VITS model that sees phonemes, not meaning, so its stress and intonation are generic. Our chunking narrows its view further: the piece "Does it handle subtle micro-breaks, like after a semicolon," doesn't reveal that it begins a question. Large cloud models read the whole sentence and plan its melody.

The engine controls *when* speech happens; it can't give a small model understanding.

## Options

| Option | Effort | Expected gain | Risk |
|---|---|---|---|
| **A. Engine-level prosody fixes:** keep questions in view, lengthen the last word before every pause, not just sentence ends | days | small | low |
| **B. Swap in a more expressive open model:** Kokoro-82M (Apache-2.0), reported to be human-like at about 2× real time on CPU ([benchmark](https://heyneo.com/blog/kokoro-tts-vs-supertonic-3-tts)) | days | large | TTFA rises to around 0.5 s; untested on this laptop |
| **C. Distillation:** generate hours of Kokoro speech and fine-tune a fast Piper voice on it (GPU via Modal credits) | weeks | potentially large: Kokoro-like prosody at Piper speed | not guaranteed to work; GPU cost |
| **D. Learned pause model:** predict pause positions and lengths from real speech (e.g. LibriTTS alignments) instead of the hand-written table | 1–2 weeks | moderate, and a genuine ML component | low: falls back to the table |

**Suggested order:** measure Kokoro on this laptop with the existing tools (naturalness, word check, TTFA, listening next to `cloud_voices/`). That answers B directly and indicates whether Kokoro is a good enough teacher for C. Option A can run in parallel.

## Engine issues found by the Coval test ([06](06-coval-comparison.md))

- **Symbols and codes** need text normalization before the model: "#" is read as "hash", "E-1047" blurs.
- **Names** (Garcia, Martinez, David, Chen) are sometimes misheard. Add them to the respelling dictionary, verified with the word check.

## Measurement questions

- **Gasping with high `noise_scale`:** heard, but not captured by the breathiness measure ([E03](../../experiments/results/e03_randomness_breathiness.txt)). Needs a better detector, or a blind listening test.
- **Coval time formatting:** does Whisper write "2.30 pm" for cloud voices on the Coval prompts too? It does on our reference paragraph ([E06](../../experiments/results/e06_cloud_pacing.txt)). Rendering a cloud voice on the Coval prompts would settle it.
- **`en_GB-cori-medium`:** a transcript dropped about 30 words. Is that the voice or Whisper?
- **Multi-speaker voices** (vctk, arctic, l2arctic, semaine) were benchmarked with speaker 0 only; other speakers may score better.

## Update (2026-09-25)

Kokoro was tested ([10](10-kokoro.md)); the listener found it natural mainly because of a soft last word before pauses ([11](11-phrase-final-softening.md)) and a livelier melody ([12](12-robotic-melody-and-well.md)). Option C, distillation, now has a written plan: [distillation-plan.md](../distillation-plan.md).
