# 10: Kokoro: a more expressive model, at a latency cost

*2026-09-25 · Code: `toast/voices.py` (`KokoroEngine`, via kokoro-onnx 0.6.1) · Data: [`benchmarks/voice_benchmark.csv`](../../benchmarks/voice_benchmark.csv) (kokoro rows) · Experiment [E08](../../experiments/e08_kokoro_vs_amy.py) ([output](../../experiments/results/e08_kokoro_vs_amy.txt))*

## Question

The remaining gap to cloud voices is prosody ([08](08-open-questions.md)). Kokoro-82M (StyleTTS2-based, Apache-2.0, 82M parameters vs Piper's ~15M) is widely described as human-like. On this laptop:

- Is it fast enough to stream?
- Is it measurably closer to the cloud references?
- Does it sound better to the listener?

## Setup

- kokoro-onnx 0.6.1 with the v1.0 model (`models/kokoro/`, 310 MB) and 28 English voices.
- Kokoro's built-in sentence and clause pauses are switched off inside our pipeline, so ToastTTS controls pausing.
- Five voices benchmarked with the standard method ([05](05-voice-benchmark.md): speed 0.9, `smart` split): the two highest-rated US female voices, a UK female voice and two US male voices.

## Results

**Speed: int8 is 10× slower, so full precision is used.**

| Model | "Well," | 5.9 s sentence | × real time |
|---|---|---|---|
| Kokoro fp32 | 228 ms | 1156 ms | 3.1–5.1× |
| Kokoro int8 | 2145 ms | 13801 ms | 0.3–0.4× (can't keep up) |
| Piper amy | 32 ms | 226 ms | 19–28× |

Dynamic int8 quantization slows this convolution-heavy model down on CPU, which is the risk noted about INT8 in the [original plan](../archive/original-plan.md) review.

**Voice benchmark** (same method as the other 39 voices):

| Voice | Natural | Wrong / 241 | TTFA (full text) | × real time |
|---|---|---|---|---|
| kokoro:af_heart | 4.37 | 0 | 284 ms | 4.1× |
| kokoro:am_puck | 4.19 | 0 | 261 ms | 4.0× |
| kokoro:bf_emma | 4.13 | 2 ("rhythm"→"rhythms" ×2) | 242 ms | 3.8× |
| kokoro:am_michael | 3.95 | 1 | 303 ms | 3.8× |
| kokoro:af_bella | 3.84 | 0 | 314 ms | 3.6× |
| *for reference:* amy-medium | 4.38 | 0 | 34 ms | 25.0× |
| *for reference:* lessac-high | 4.44 | 0 | 184 ms | 4.1× |

**Pacing vs the cloud reference** (E08, Whisper word timestamps as in [07](07-cloud-reference-and-pacing.md)):

| Recording | Natural (20 s) | wpm talking | `,` pause | `.` pause |
|---|---|---|---|---|
| Soniox Grace (reference) | 4.47 | 195 | 440 ms | 800 ms |
| Kokoro af_heart, its own pacing (whole paragraph) | 4.36 | 212 | 360 ms | 570 ms |
| **Kokoro af_heart 1.0× through ToastTTS** | 4.33 | **194** | 440 ms | 810 ms |
| Kokoro af_heart 1.1× through ToastTTS | 4.30 | 202 | 400 ms | 760 ms |
| amy 1.2× through ToastTTS (current default) | 4.43 | 178 | 500 ms | 800 ms |

**Streaming** (E08, fake LLM, TTFA from the first token, median of 3; stalls from the worst run):

| Voice | 30 tokens/s | 10 tokens/s |
|---|---|---|
| Kokoro af_heart 1.0× | 344 ms, 0 stalls | 468 ms, **1 stall (171 ms)** |
| amy 1.2× | 97 ms, 0 stalls | 223 ms, 0 stalls |

## Findings

- **Through ToastTTS, Kokoro at its natural speed matches Soniox's pacing almost exactly:** 194 vs 195 wpm while talking, the same comma and period pauses. Left to its own pacing, Kokoro rushes (212 wpm) with shorter sentence pauses (570 ms), so our pacing layer improves Kokoro too.
- **Predicted naturalness doesn't separate Kokoro from amy** (4.30–4.43 on every render). UTMOS rates overall naturalness per clip; it evidently doesn't capture the stress and intonation differences the listener described. **This comparison must be settled by ear.**
- **Latency cost:** about 0.35 s to first audio, 3–4× worse than amy. That's still in the range of cloud services' measured TTFA (ElevenLabs v3 320 ms, Soniox 255 ms, both including network), but with **one stall at 10 tokens/s**: only 2.4× real time end to end, so a slow LLM can starve playback.

## Pending

- The listener's verdict on `out/listen/20_*`, `21_*` vs `16_amy_1.2x.wav` and `cloud_voices/`.
- If Kokoro wins by ear: fix the 10 tokens/s stall. Options are a larger "running low" margin for slower voices, or making the threshold depend on the voice's speed.
- If it's close: Kokoro becomes the **teacher** candidate for distillation into a fast Piper student ([08](08-open-questions.md) option C).

→ Back to [08: open questions](08-open-questions.md)
