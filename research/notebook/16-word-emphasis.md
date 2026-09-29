# 16: Word emphasis: let the LLM decide what matters

*2026-09-26 · Code: `toast/emphasis.py`, [`scripts/emphasis_demo.py`](../../scripts/emphasis_demo.py) · Results: [subtle](../../experiments/results/emphasis_demo.txt), [strong](../../experiments/results/emphasis_demo_strong.txt) · Audio: `out/listen/emphasis/`*

## Question

A small TTS sees only words, not meaning, so it cannot know which word carries contrastive focus — yet stressing the wrong word changes the entire sentence meaning.

## Idea: the LLM decides, the engine performs

The LLM writing the reply understands the meaning, so it marks the focus word with asterisks: `I didn't say *he* stole it.` ToastTTS performs the emphasis on that word the way speakers mark focus:
- a pitch accent over the word;
- lengthening;
- extra loudness;
- (strong preset) **post-focus compression**: the words after the focus get a flatter, lower melody.

All of it is done with Praat PSOLA, which keeps the voice.

**Where the words are:** Piper can report per-phoneme durations if its ONNX has an alignment output. Piper ships a tool that adds it to any voice without retraining:

```
python -m piper.patch_voice_with_alignment models/en_US-amy-medium.onnx --output models/aligned/en_US-amy-medium.onnx
```

Word boundaries are the space phonemes. If espeak splits words differently from the text (e.g. numbers), the emphasis is skipped rather than guessed.

## Results: "I didn't say he stole it.", amy at 1.2×

| Preset | Accent | Longer | Louder | Post-focus | Measured rise on the marked word | Wrong words |
|---|---|---|---|---|---|---|
| subtle | +22% pitch | ×1.18 | +3 dB | none | +2.3 to +4.0 st | 0 in all 7 versions |
| **strong** | +42% pitch | ×1.40 | +4.5 dB | range ×0.45, −1.5 st | **+4.7 to +6.6 st** | 0 in all 7 versions |

The sentence-final "it." can't be measured: creaky voice at the end defeats the pitch tracker (a false +31.7 st in the subtle run, no value in the strong run). It has to be judged by ear.

**Listening:**
- **subtle:** judged barely noticeable — too weak to carry meaning reliably.
- **strong:** judged clearly noticeable. The strong preset is the default for integration.

## Status

A working prototype, not yet part of the streaming engine. Integration means:
- markers survive the chunker;
- a Piper voice with the alignment output is used;
- the LLM is prompted to mark contrastive focus.

**Emotion** is a natural next step on the same machinery, at sentence level: pitch level and range, rate and energy presets from LLM tags. The alternative is an emotion-conditioned student trained on DailyTalk's per-utterance emotion labels ([08](08-open-questions.md)).
