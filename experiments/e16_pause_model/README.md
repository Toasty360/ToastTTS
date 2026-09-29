# e16_pause_model — learned phrase-break / pause-duration model

Tiny text-in → pauses-out model for ToastTTS. Predicts where pauses occur in a
sentence and how long they last, so ToastEngine can chunk and pace synthesis
with learned timings instead of hand-written punctuation rules.

## What it is

- Input: raw text. Output per word: pause probability and predicted silence
  (ms) after the word.
- Model: word embedding (24d) + 4 cheap token features + 2-layer BiLSTM
  (48 hidden) + binary pause head. **93,605 parameters**, trains on CPU in
  ~15 seconds, inference is milliseconds.
- Pause duration is **not** learned by the network: the duration-bucket head
  collapsed to a single bucket, so durations come from an empirical lookup —
  median observed gap per punctuation kind in the training data
  (comma → 440 ms, sentence-final `.`/`!`/`?` → 720 ms).

## Data

- LibriSpeech `dev-clean` (CC-BY 4.0), longest-first subset, speaker-disjoint
  train/val split: 111 train files / 74 val files.
- Word timestamps from local Whisper `small.en` (CPU, ~2–3x realtime).
  No cloud APIs, no paid services.
- Usable word-span audio: **33.6 min train / 24.7 min val**
  (5,758 / 4,090 words; 5,647 / 4,016 word boundaries).
- Pause label: inter-word gap ≥ 150 ms → pause (9.3% positive rate).

## Results (held-out val, speaker-disjoint)

| metric | value |
|---|---|
| pause precision | 0.900 |
| pause recall | 0.972 |
| pause F1 | 0.934 |
| tp / fp / fn | 378 / 42 / 11 |

Example:

```
IN:  She bought apples, oranges, and bananas. Then she left.
OUT: She bought apples, [pause: 440ms] oranges, [pause: 440ms]
     and bananas. [pause: 719ms] Then she left.
```

## How to run

```bash
~/workspace/.pause-venv/bin/python transcribe.py --files data/subset1h/files_train.txt --outdir data/words_train
~/workspace/.pause-venv/bin/python build_labels.py --train-json data/words_train --val-json data/words_val --outdir data/labels
~/workspace/.pause-venv/bin/python train.py --train data/labels/examples_train.jsonl --val data/labels/examples_val.jsonl --out data/model
~/workspace/.pause-venv/bin/python predict.py --checkpoint data/model/pause_model.pt --text "Hello world, how are you?"
```

For ToastEngine integration:

```python
from predict import PausePredictor
p = PausePredictor("data/model/pause_model.pt")
p.predict_pauses("Hello world, how are you?")
# [{"word": "Hello", "pause_prob": 0.004, "pause_ms": 0}, ...]
```

## Limitations (read before using)

1. **Pauses in this data are punctuation pauses.** 99.2% of observed pauses
   follow `,` or `.`; the corpus files average ~5 s (single sentences), so
   mid-sentence pauses without punctuation are nearly absent. The model does
   not predict phrase breaks in unpunctuated text — that needs longer-form
   or conversational data (audiobooks with chapter-length passages,
   podcasts, dialogue).
2. **Punctuation inventory is commas and periods.** The training set contains
   zero `:`/`;` tokens and two each of `?`/`!`. Colons get no pause; this
   mirrors the data, not ideal prosody.
3. **Durations are medians, not predictions.** Comma/period gap distributions
   overlap heavily (comma p25–p75: 320–580 ms; period: 500–1020 ms), so the
   classifier could not separate them; the lookup captures the central
   tendency (440 vs 720 ms).
4. **Whisper word timestamps are approximate.** Many true micro-pauses show
   as 0 ms gaps; the model learns from the clear cases.
5. Training is CPU-only by design; the whole pipeline (minus transcription)
   runs in under a minute.

## Next steps

- Train on longer-form speech (chapter-length audiobook passages or
  conversational corpora) to capture non-punctuation phrase breaks.
- Add syntactic features (POS tags, dependency depth) so the model can
  generalize beyond punctuation.
- Wire `predict_pauses` into ToastEngine's chunker, replacing fixed
  punctuation rules; A/B by listening test against the current engine.
