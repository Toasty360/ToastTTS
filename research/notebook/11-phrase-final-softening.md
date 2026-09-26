# 11: The soft last word: what made Kokoro sound natural

*2026-09-25 · Experiments [E09](../../experiments/e09_phrase_final_softening.py) ([output](../../experiments/results/e09_phrase_final_softening.txt)), [E10](../../experiments/e10_softening_amy.py) ([output](../../experiments/results/e10_softening_amy.txt)) · Code: `toast/pacing.py` `soften_ending()`, `toast/text_normalize.py`*

## Finding: the soft last word

Listening to Kokoro ([10](10-kokoro.md)) identified the key difference: Kokoro softens the last word's amplitude before a natural pause, where amy stays loud. That single difference accounts for much of the naturalness gap. This matches the project's opening idea (bring the volume down before a pause instead of stopping abruptly), now observed in a model that does it naturally.

Kokoro was also heard pausing inside "3:15 PM". Kokoro's own behaviour was left alone as out of scope (D25). Only one general fix was kept: times are written as words before synthesis ("three fifteen PM"), which removed the 480 ms pause Kokoro put between "3" and "15" and is harmless for every voice (`text_normalize.py`, tests). Kokoro's remaining pause after "PM" (360–580 ms with any spelling of PM) was left alone. The cloud voices and amy don't pause inside "3:15 PM" at all.

## E09: measuring the soft last word

Whisper word timestamps locate every word followed by a pause of 150 ms or more. Two numbers per pause (medians):

- **drop:** the last word's loudness vs the 1.5 s of speech before it.
- **fade:** the last 30% of that word vs its first 70%.

| Recording | Pauses | Drop | Fade |
|---|---|---|---|
| Soniox Grace | 16 | −0.3 dB | −3.2 dB |
| Deepgram Aura-2 Thalia | 17 | +1.2 dB | −1.0 dB |
| Kokoro af_heart, own pacing | 19 | −1.0 dB | −3.6 dB |
| Kokoro af_heart through ToastTTS | 17 | −0.7 dB | −3.8 dB |
| **amy 1.2× through ToastTTS** | 14 | **+1.1 dB** | −2.7 dB |

The direction matches what was heard: Kokoro and Soniox get softer into a pause, while amy's last word is *louder* than the speech before it. (Deepgram Thalia also doesn't soften by this measure.)

## Implementation: `soften_ending()`

Over the last `SOFT_TAIL_MS` of every piece, which is roughly the last word since every piece is followed by a pause, the volume eases down smoothly (raised cosine) to `SOFT_TAIL_DB`. It runs before the 12 ms edge fades, and costs nothing measurable (live TTFA 94 ms, no stalls).

## E10: calibration attempt, and its limit

Settings were swept with E09's measure, 2 renders each:

| Tail | Depth | Drop | Fade |
|---|---|---|---|
| off | — | +1.2 | −3.9 |
| 300 ms | −4 dB | +0.8 | −2.8 |
| 400 ms | −4 dB | +0.7 | −3.4 |
| 400 ms | −6 dB | +0.9 | −3.2 |
| **500 ms** | **−6 dB** | +0.7 | −4.8 |
| 500 ms | −9 dB | +0.4 | −4.9 |

**The measure is too noisy to calibrate with.** The same "off" setting gave fade −2.9 dB in a first sweep and −3.9 dB here: about ±1 dB between runs, as large as the effect being tuned. Two reasons: Whisper's word boundaries are only approximate, and a 400–500 ms ramp also lowers the end of the *previous* word, which moves the "before" reference too. The measure shows the direction; **the setting is chosen by ear.**

Default for now: **500 ms, −6 dB** (closest to Kokoro in the first sweep).

Listening set, identical pauses (same seed), so softening is the only difference:
- `out/listen/30_amy_1.2x_no_softening.wav`
- `out/listen/31_amy_1.2x_soft_400ms_-4dB.wav`
- `out/listen/32_amy_1.2x_soft_500ms_-6dB_default.wav`
- `out/listen/33_amy_1.2x_soft_500ms_-9dB.wav`

## Pending

The listener's choice among 30–33. A better measure for future tuning would align words with forced alignment rather than Whisper timestamps, or work at the phoneme level.

→ Back to [08: open questions](08-open-questions.md)
