# ToastTTS

**Natural pauses and a fast start for on-device text-to-speech.**

Small TTS models run fast on a laptop but speak like they're reading a list. ToastTTS sits between an LLM and a small TTS model (Kokoro or Piper). It splits the reply at natural pause points as it streams in, inserts human-like pauses with smooth fades, and starts speaking before the reply is finished. It can be interrupted mid-reply and reports what the listener heard, for full-duplex voice assistants.

- **About 0.1 s** from first LLM token to first sound (89–104 ms measured at 30 tokens/s), with no playback stalls
- **5.1–6.3% WER** on a replication of the [Coval](https://github.com/coval-ai/benchmarks) cloud TTS benchmark, among commercial services such as Cartesia Sonic and ElevenLabs Flash
- **CPU only, runs locally.** Two tuned voices: **daniel** (Kokoro, default, about 7× faster than real time) and **amy** (Piper, about 25–60× faster, the numbers below)

## Results at a glance

Cloud TTS comparison (Coval `tts-v1` prompts, scored by OpenAI `whisper-1` as Coval does; cloud rows from the Coval board, 2026-09-08):

| Model | WER | Median TTFA | Runs on |
|---|---|---|---|
| ElevenLabs Eleven v3 Conversational | 4.3% | 320 ms | cloud |
| Deepgram Aura-2 | 5.0% | 290 ms | cloud |
| Cartesia Sonic 3.6 | 5.3% | 440 ms | cloud |
| **ToastTTS + amy-medium** (fast profile) | **5.9%** | **97 ms** | laptop CPU |
| ElevenLabs Flash v2.5 | 6.5% | 185 ms | cloud |

Cloud TTFA includes the network round trip; ToastTTS's does not. About 3.5 WER points come from how Whisper writes times ("2.30 pm"). Full table, method and caveats: [research/notebook/06](research/notebook/06-coval-comparison.md).

All findings, methods and decisions: **[research/](research/README.md)**.

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Windows commands shown.

```
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python -m piper.download_voices --download-dir models en_US-amy-medium
.venv\Scripts\python scripts\live.py --play
```

On macOS/Linux, use `.venv/bin/python` instead of `.venv\Scripts\python` and `/` for paths (`brew install uv` to get uv):

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m piper.download_voices --download-dir models en_US-amy-medium
.venv/bin/python scripts/live.py --play
```

The default voice, **daniel** (Kokoro bm_daniel at 1.2×), needs the Kokoro model files ([notebook/10](research/notebook/10-kokoro.md)) in `models/kokoro/`:

```
curl -L --create-dirs -o models/kokoro/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx
curl -L --create-dirs -o models/kokoro/voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin
```

Use `--voice amy` for the fast Piper voice, or any other voice name such as `kokoro:af_heart`. Profiles (speed and streaming margin per voice) live in `PROFILES` in [`toast/engine.py`](toast/engine.py).

`live.py` streams a reply from a simulated LLM (30 tokens/s), plays it live, and prints when each piece was cut plus the TTFA. Options include `--tokens-per-second 10`, or `--text samples\no_early_comma.txt`.

### Speak your own text

```
.venv\Scripts\python scripts\say.py                                   # type lines, hear them
.venv\Scripts\python scripts\say.py "Well, hi there! How can I help?"
some_llm_cli "tell me a joke" | .venv\Scripts\python scripts\say.py -  # speaks while text streams in
```

Add `--save file.wav` to keep the audio, or `--no-play` to only generate it. Each run prints the TTFA and how the text was cut.

### Use it from code

```python
from toast.engine import ToastEngine

tts = ToastEngine()                      # daniel at 1.2x; ToastEngine("amy") for the fast voice
result = tts.say("Well, let me check that for you.")   # plays live
print(result.ttfa_ms, result.pieces)

tts.say(token for token in llm_stream)   # any iterable of text: speaks as it arrives
audio = tts.synthesize("Any text.")      # float32 numpy array at tts.sample_rate
tts.save("Any text.", "out.wav")
```

In a full-duplex assistant (ASR → LLM → ToastTTS), speak in the background and cut off on barge-in:

```python
tts = ToastEngine(threads=4, open_audio=True)   # leave CPU cores for the ASR and the LLM
reply = tts.speak(token for token in llm_stream) # returns at once
...                                              # user starts talking:
reply.stop()                                     # fades out within ~20 ms, stops generating
history.append({"role": "assistant", "content": reply.heard()})  # only what was played
```

`reply.done` says when it finished playing; `reply.wait()` blocks and returns the timings. Cancelling the LLM stream is up to the caller.

## Repository map

| Path | Contents |
|---|---|
| [`toast/`](toast/) | The engine: `engine` (the public entry point, `ToastEngine`), `text_normalize`, `stream_chunker` (where to cut), `pacing` (pauses, fades), `voices` (one interface for Piper/Kokoro/Kitten), `metrics`, `word_check` (Whisper), `naturalness` (UTMOS), `speech_stats` |
| [`scripts/`](scripts/) | Tools: `say.py` (speak any text: typed, argument or piped), `audio_check.py` (checks whether the speakers swallow the first word), `live.py` (streaming demo), `render.py` (recordings + timing), `bench_voices.py` (all voices), `bench_coval.py` (cloud comparison), `find_voices.py` (multi-speaker scan) |
| [`research/`](research/README.md) | Write-up: lab notebook 01–16, methodology, decision log, original plan |
| [`experiments/`](experiments/README.md) | E01–E19: re-runnable investigations with saved outputs |
| [`benchmarks/`](benchmarks/README.md) | All measured data (CSV), documented |
| [`samples/`](samples/README.md) | Test texts, including Coval's prompts (Apache-2.0) |
| [`cloud_voices/`](cloud_voices/README.md) | Cloud reference recordings (listening and measurement only) |
| [`tests/`](tests/) | Chunker and word-check tests (`.venv\Scripts\python -m pytest`) |
| `models/`, `out/` | Downloaded voices and generated audio (not tracked) |

## Reproducing the benchmarks

```
.venv\Scripts\python scripts\bench_voices.py                  # 38 voices + Kitten (~25 min, needs all en_* voices)
.venv\Scripts\python scripts\bench_coval.py --asr whisper-1   # needs OPENAI_API_KEY, about $0.17
.venv\Scripts\python experiments\e05_word_clarity_by_voice.py # any single experiment
```

## Status

Pacing, streaming and measurement are working. Stage-2 of the ex02 voice fine-tune is complete: comma pauses measured at p50 140 ms (down from 330 ms in stage 1), 11 word errors versus amy's 12 and UTMOS naturalness of 4.25 versus amy's 4.35 on the standard 40-sentence eval, with all six pre-registered pipeline checks passed (method and metrics in [research/stage2_pause_control.md](research/stage2_pause_control.md); checkpoint pick in [experiments/e19_stage2_ab](experiments/e19_stage2_ab/README.md), verdict pending blind listening). The open problem is still **prosody**: stress, pre-pause shaping and question intonation, which the cloud voices have and a 15M-parameter model doesn't. Next steps (a more expressive model such as Kokoro, distillation, a learned pause model) are in [research/notebook/08](research/notebook/08-open-questions.md).
