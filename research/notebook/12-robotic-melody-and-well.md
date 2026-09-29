# 12: Why amy sounds robotic: a flat melody, and a "Well," that sounds final

*2026-09-25 · Experiments [E11](../../experiments/e11_pitch_profile.py) ([output](../../experiments/results/e11_pitch_profile.txt)), [E12](../../experiments/e12_well.py) ([output](../../experiments/results/e12_well.txt)) · Code: `toast/prosody.py`, `toast/pacing.py` (`OPENER_PAUSE_MS`, softening cap), `toast/stream_chunker.py` (`OPENERS`, `attach_openers`)*

## Question

Even with the soft last word ([11](11-phrase-final-softening.md)), amy was heard as robotic next to af_heart. A voice has two layers:

- **Timbre:** what the voice sounds like. Voice conversion could transfer it, but it isn't what makes speech robotic.
- **Prosody:** melody, stress and rhythm. This is where "robotic" usually lives, and it can be measured and partly edited.

## E11: amy's melody is flat

Pitch was measured with Praat (`toast/prosody.py`), in semitones:

| Recording | Median pitch | Spread (SD) | Range (5–95%) | Movement |
|---|---|---|---|---|
| Soniox Grace | 164 Hz | 3.96 st | 13.02 st | 35.8 st/s |
| Deepgram Thalia | 210 Hz | 3.71 st | 10.56 st | 33.0 st/s |
| Kokoro af_heart | 200 Hz | 3.87 st | 10.47 st | 45.8 st/s |
| **amy 1.2×** | 197 Hz | **2.71 st** | **6.84 st** | **28.1 st/s** |

amy uses about **two-thirds of the pitch range** of the natural voices, and its pitch moves more slowly. That's a measurable basis for "robotic".

**Transfer attempt:** amy's pitch movement was scaled around its own median (Praat PSOLA) by the ratio of spreads (×1.43 toward af_heart, ×1.46 toward Grace):
- Range rose from 6.84 to 9.2–9.4 st, close to the others.
- Spread rose only from 2.71 to 2.88 st, and movement *fell* slightly (28 → 26 st/s).
- The edit is therefore partial: it widens the melody but doesn't make it livelier. The spread-based factor was probably inflated by pitch-tracking outliers; matching the range would be more robust.
- No damage: 0 wrong words, naturalness 4.41–4.43 vs 4.46.

Listening files: `out/listen/40_amy_pitch_as_is.wav`, `40_amy_pitch_like_af_heart_x1.43.wav`, `40_amy_pitch_like_grace_x1.46.wav`. Listening verdict pending.

## E12: the "Well," contour differs between voices

**Is it the pronunciation?** One hypothesis was that amy says "wl" where the others say "wɛl". Both models receive the **same phonetic spelling**, `wˈɛl`. The difference is in how the word is performed: vowel length and pitch.

**Pitch shape** (Whisper word boundaries at the very start of a file include leading silence, so the loudness and duration figures in E12 are unreliable; pitch shape is usable):
- **Kokoro:** rise then fall (−1.0 → +1.9 → −1.9 st), which sounds like "…and I'm continuing".
- **amy:** a straight fall (+1.4 → 0 → −1.8 st), the shape of a finished statement. Whisper transcribed amy's isolated "Well," as **"Well."**; every other voice was transcribed "Well,".
- **Cause:** the `smart` split makes "Well," a one-word piece spoken in isolation, so the model gives it a final contour. The phrase-final softening then faded half of that single word.

**The pause after it** (Whisper word gaps, same method for every voice):

| Recording | Pause after "Well," | After "honest," |
|---|---|---|
| Soniox Grace | 320 ms | 460 ms |
| Deepgram Thalia | 180 ms | 260 ms |
| Deepgram Hannah | 80 ms | 400 ms |
| Kokoro af_heart | 260 ms | 380 ms |
| amy, before | **420 ms** | 480 ms |

Natural voices pause *less* after "Well," than after other commas; amy paused more.

## Changes

1. **Openers** ("Well,", "So,", "Okay,", "Actually,"…; `OPENERS`) get a shorter configured pause: 60–140 ms instead of 180–260 ms. Measured the same way, amy's pause after "Well," went from 420 to 360 ms.
2. **Softening is capped at the last 30% of a piece,** so a one-word piece isn't faded through its middle.
3. **Option `attach_openers`** (`say.py --attach-openers`): keeps "Well," with the following words, so the model says it knowing more follows. Measured pause 280 ms, inside the natural range, and the intonation is no longer isolated. **Cost:** the first piece waits for more text.

| | TTFA at 30 tokens/s | TTFA at 10 tokens/s |
|---|---|---|
| Openers separate (default) | 94 ms | 228 ms |
| `attach_openers` | 255 ms | 662 ms |

Files: `out/listen/51_amy_well_separate_short_pause.wav`, `52_amy_well_attached.wav`. **Listener judgment:** both renders were judged good, with no clear preference. Since attaching costs about 160 ms of start time for no clear audible gain, **the default stays separate** (with the shorter opener pause and the softening cap); `--attach-openers` remains available.

## What this implies

Both findings point to the same root cause: **a small model's generic prosody.** Post-processing can widen the melody and fix pauses, but it can't give amy Kokoro's rise-fall "Well," or its stress. That is the case for distillation ([08](08-open-questions.md) option C): teach a fast Piper voice Kokoro's prosody by training it on Kokoro's speech.

→ Back to [08: open questions](08-open-questions.md)
