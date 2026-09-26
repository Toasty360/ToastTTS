# 02: Natural pauses have to be inserted explicitly

*2026-09-25 · Experiments [E02](../../experiments/e02_model_pause_gaps.py), [E03](../../experiments/e03_randomness_breathiness.py), [E04](../../experiments/e04_piece_end_punctuation.py)*

## The idea

The starting idea: at the end of each piece of audio, *fade the volume down* instead of cutting to digital zero, so pauses sound natural. Two existing audio practices back this up:

- **Fades prevent clicks.** Stopping mid-waveform produces an audible pop, and a 5–20 ms fade removes it.
- **Real pauses aren't silent.** Every recording has faint room hiss. Podcast editors fill gaps with "room tone" so pauses don't sound dead.

## First attempt: stretch the model's own pauses (failed)

If the model already leaves gaps at punctuation, they could be found in the audio and lengthened, keeping whole-sentence intonation intact. **E02** checked for 10 ms frames below -40 dB, keeping runs of 30 ms or more:

| Voice | Commas/;/: in the text | Gaps found | Typical gap |
|---|---|---|---|
| Piper libritts_r 3922 | 5 | 5 | 30–50 ms |
| Kitten micro | 5 | 16 | 30–110 ms, spread across words |
| Piper amy-medium | 5 | 17 | 30–340 ms, spread across words |

([full output](../../experiments/results/e02_model_pause_gaps.txt))

- 3922 barely pauses at commas at all, which is why it sounded "rushed".
- Kitten and amy leave gaps between ordinary words (and stop consonants), so real pauses can't be told apart.

**Conclusion:** pauses can't be reliably found in the audio. The engine inserts its own.

## Implementation (`toast/pacing.py`)

For each piece of text:

1. **Synthesize** it. Pieces ending mid-sentence in `;` `:` or a dash are sent to the model ending in `,` instead ("more is coming").
2. **Trim** the model's own leading/trailing silence: frames below -45 dB relative to the peak. Keep 10 ms before the speech (silence there is heard as delay) and 25 ms after it (soft final consonants).
3. **Fade** 12 ms in and out with a sin² curve.
4. **Pause.** The length depends on the punctuation, uniformly random within the range:

   | After | `,` | `;` | `:` | `—` | `...` | `.` | `!` | `?` | no punctuation |
   |---|---|---|---|---|---|---|---|---|---|
   | ms | 180–260 | 300–380 | 280–360 | 260–360 | 500–700 | 480–620 | 450–580 | 520–680 | 60–120 |

5. **Room tone** runs continuously *under everything* at -60 dBFS: white noise with the highs smoothed off, pre-generated and looped. If it only played during pauses, you'd hear it switch on and off.
6. **Phrase-final softening** (added later, [11](11-phrase-final-softening.md)): the last 500 ms of every piece eases down to −6 dB, so the last word settles into the pause.
7. **Phrase-final lengthening.** The last piece of a sentence that was split into several pieces is spoken at 0.94× speed, since people slow down at the end of a sentence.

Checked by measurement: every one of the 19 planned pauses was present (180–700 ms), and the room tone sat at -60.1 dB.

## Where to cut: three split modes

| Mode | Cuts at | Result |
|---|---|---|
| `clause` | every comma, semicolon, dash, sentence end | most pause control, choppiest intonation |
| `sentence` | sentence ends only | smoothest intonation, but slow start and no pauses inside lists |
| `smart` | sentence ends, `; : — ...`, and commas **only after a short piece** (≤ 4 words: "Well,", "12,", "rhythm,") | the chosen default |

How the listening rounds went:

- Kitten in `sentence` mode was preferred at first.
- But it read "12, 45, and 108" without pauses, which led to `smart`.
- `smart` pauses where people do (asides and list items) and keeps long clauses together, so the model keeps its intonation across them. Both `smart` Piper renders were judged "good".

The split rules later moved into the streaming chunker, so batch and live use one implementation (see [03](03-streaming.md)).

## Voice character: Piper's randomness settings

The 3922 voice file ships with `noise_scale = noise_w = 0.333`, half of Piper's usual 0.667 / 0.8. The listener described it as sounding "not interested", with pacing "fast".

- `noise_scale` controls randomness of the sound itself (more expressive).
- `noise_w` controls randomness of timing (a less robotic rhythm).

| Setting | Listening result |
|---|---|
| 0.333 / 0.333 (file default) | flat, bored |
| **0.333 / 0.8** (calm sound, uneven rhythm) | good |
| **0.5 / 0.8** | good |
| 0.667 / 0.8 ("lively") | "gasping", weird breathing |

**Gasping: cause unresolved.** The hypothesis was that high `noise_scale` brings out breath noise learned from audiobook training data. **E03** counted noise-like, audible frames over the paragraph (3 runs each): 329 (default), 356 (0.333/0.8), 360 (0.5/0.8), 354 (lively) ([output](../../experiments/results/e03_randomness_breathiness.txt)). Lively is *not* worse than the settings that sounded fine. So either this measure can't capture the gasps (it also counts "s", "f", "sh"), or the cause is something else. An early single-run check had suggested 15–40% more breathy frames for lively; this repeated measurement doesn't confirm it. Practical outcome: keep `noise_scale` ≤ 0.5.

## A correction: the ";" fix has weak evidence

"semicolon;" sounded swallowed at the end of a piece. A single-run comparison suggested Piper rushes the final word before `;` (2.86 s) but not before `,` (3.18 s), so mid-sentence endings are now sent to the model as `,`.

**E04** repeated this with 6 runs per ending: `;` 2.94 ± 0.23 s, `,` 2.98 ± 0.21 s, `.` 2.99 ± 0.17 s ([output](../../experiments/results/e04_piece_end_punctuation.txt)). The difference is within run-to-run noise, so **the original single-run result was noise**. The change is harmless (our code adds the pause either way), so it stays, but it doesn't explain the problem. The real cause was the voice model itself: see [04](04-voice-intelligibility.md).

*Lesson recorded:* Piper output is random on every call. Any claim about its output needs repeated runs.

→ Next: [03: streaming](03-streaming.md)
