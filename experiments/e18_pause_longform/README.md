# e18_pause_longform — pause model retrained on chapter-length audiobooks

Retrain of the e16 pause model on **long-form** speech. e16's training clips
averaged ~5 s (single sentences), so 99.2% of its pauses followed `,` or `.`
and it learned punctuation→pause, not real phrase breaks. This run uses
chapter-length LibriVox audiobook passages (20–60 min each) where narrators
pause mid-sentence at phrase boundaries with no punctuation in sight.

## Data (all public domain, LibriVox via archive.org)

Train (~4 h, 5 narrators):
- Moby Dick (Stewart Wills): ch3, ch4-7
- Sherlock Holmes (Mark F. Smith): ch1
- Frankenstein (Caden Vaughn Clegg): ch0, ch3
- Jane Eyre: ch2, ch3
- Pride and Prejudice: ch1-3, ch4-5

Validation (~1.5 h, held-out narrator):
- Heart of Darkness (Kristin Luoma): ch1a, ch1b

## Pipeline (same as e16, scripts copied)

1. `transcribe.py` — Whisper `small.en` word timestamps per chapter
   (patched: output naming works for `.mp3`, not just `.flac`).
2. `build_labels.py` — inter-word gaps → pause labels (≥150 ms).
3. `analyze_gaps.py` — **the** diagnostic: % of pauses NOT after
   punctuation (e16: 0.8%; want much higher here).
4. `train.py` — same 93k-param BiLSTM, 25 epochs, CPU.

## Success criteria

- Non-punctuation pauses are a substantial share of labels (else the data
  didn't fix the problem — say so honestly).
- Validation F1 on non-punctuation pauses reported separately.
- A/B/C listening: pause table vs e16 vs e18 on unpunctuated phrase breaks.

## Results (2026-09-26) — the run failed, documented honestly

**Transcription** worked: 11 chapters via Whisper `small.en` on a Modal T4,
~22 min wall clock at 0.06x real-time (vs 1.04x on local CPU).

**Gap analysis** (the key diagnostic): only **1.4%** of pauses ≥150 ms are
non-punctuation (89 of 6,430; train 1.9%, val 0.2%), vs 0.8% in e16.
Audiobook narrators still pause overwhelmingly at punctuation — the long-form
data did not provide the phrase-break signal this run was designed to capture.

**Training**: val P=0.23 / R=0.76 / **F1=0.354** (e16: 0.934). The model did
not learn placement: pause probabilities sit at 0.40–0.63 for nearly every
word, i.e. maximum uncertainty around the 0.5 decision threshold. At
inference it predicts a 740 ms pause after almost every word, which is
unusable. Learned duration medians: comma 360 ms, sentence stop 640 ms.

**A/B/C**: rendered (abc.py, 18 wavs) but the C samples are audibly broken
(pause after every word); no listening verdict requested.

**Interpretation**: with 98.6% of pauses following punctuation and five
varied narrators, the text→pause mapping in this data is near-ambiguous for
a word-identity model — it hedged toward "pause often" under the recall-heavy
class weight instead of learning placement. Conversational speech
(disfluencies, unpunctuated phrase breaks) would be the data to try next,
not more audiobooks.

**Status**: e18 checkpoint kept for reproducibility only. e16 remains the
better learned model; the hand-written table remains the default.
