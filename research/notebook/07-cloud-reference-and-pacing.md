# 07: Matching cloud voices' pacing

*2026-09-25 · Experiment [E06](../../experiments/e06_cloud_pacing.py) · Code: `toast/speech_stats.py` · Reference audio: [`cloud_voices/`](../../cloud_voices/)*

## Question

The listener's verdicts after the benchmark were: ryan "doesn't sound natural"; amy "is good but needs about 1.1× speed". Rather than guess a speed, can we measure how cloud voices pace the same text and tune toward it?

## Setup

The author provided three cloud recordings of `samples/reference.txt`: Deepgram Aura-2 (Thalia), Deepgram Flux (Hannah, generated at 1.1× speed) and Soniox TTS (Grace). All are 24 kHz mono. They're used here only as a measurement reference ([D18](../decisions.md)).

`toast/speech_stats.py` uses Whisper word timestamps to measure:
- **Words per minute:** overall, and while talking (pauses of 80 ms or more removed).
- **Pause lengths:** grouped by the punctuation before them.

amy was rendered through our full pipeline at four speeds and measured the same way.

## Results ([output](../../experiments/results/e06_cloud_pacing.txt))

| Recording | Natural (first 20 s) | wpm overall | wpm talking | Pause after `,` | Pause after `.` |
|---|---|---|---|---|---|
| Soniox Grace | 4.47 | 154 | 195 | 440 ms | 800 ms |
| Deepgram Aura-2 Thalia | 4.46 | 144 | 188 | 380 ms | 800 ms |
| Deepgram Flux Hannah (1.1×) | 3.75 | 145 | 183 | 500 ms | 910 ms |
| ToastTTS amy 0.9× | 4.34 | 122 | 152 | 520 ms | 820 ms |
| ToastTTS amy 1.0× | 4.39 | 128 | 162 | 580 ms | 800 ms |
| ToastTTS amy 1.1× | 4.47 | 134 | 166 | 440 ms | 830 ms |
| **ToastTTS amy 1.2×** | 4.25 | 141 | **179** | 460 ms | 840 ms |

Pause values are medians. Naturalness on a single 20 s clip varies by about ±0.15 between runs; an earlier run of the same comparison gave 4.48 for 1.1× and 4.31 for 1.2×.

## Findings

- **Our pause lengths already match the cloud voices** when measured the same way (commas ~440–520 ms vs 380–500; periods ~800–840 vs 800–910). They looked short on paper (the configured comma pause is 180–260 ms) only because Whisper's gaps also include the quiet edges of words.
- **The gap was speaking rate.** amy at 0.9× talked at 152 wpm against 183–195 for the cloud voices, which confirms the listener's "needs to be faster". At 1.2× amy reaches 179 wpm.
- **Predicted naturalness is in the same range as Soniox and Deepgram Thalia** at every speed. The measurable differences are now smaller than the metric's noise, so the remaining gap has to be judged by ear ([08](08-open-questions.md)).
- Whisper writes "3.15 pm" for the cloud recordings too, which is relevant to the Coval time-format question ([06](06-coval-comparison.md)).

## Decision

By ear, 1.2× was "nice". **Default: amy-medium at 1.2×** ([D16](../decisions.md)). ryan was dropped despite its best Coval WER.

→ Next: [08: open questions](08-open-questions.md)
