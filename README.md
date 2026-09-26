# ToastTTS

**Natural pauses and a fast start for on-device text-to-speech.**

Small TTS models run fast on a laptop but speak like they're reading a list. ToastTTS sits between an LLM and a small TTS model (Piper). It splits the reply at natural pause points as it streams in, inserts human-like pauses with smooth fades and room tone, and starts speaking before the reply is finished.

- **About 0.1 s** from first LLM token to first sound (89–104 ms measured at 30 tokens/s), with no playback stalls
- **5.1–6.3% WER** on a replication of the [Coval](https://github.com/coval-ai/benchmarks) cloud TTS benchmark, among commercial services such as Cartesia Sonic and ElevenLabs Flash
- **CPU only, runs locally.** The default voice is generated about 25× faster than real time

## Results at a glance

Cloud TTS comparison (Coval `tts-v1` prompts, scored by OpenAI `whisper-1` as Coval does; cloud rows from the Coval board, 2026-09-08):

| Model | WER | Median TTFA | Runs on |
|---|---|---|---|
| ElevenLabs Eleven v3 Conversational | 4.3% | 320 ms | cloud |
| Deepgram Aura-2 | 5.0% | 290 ms | cloud |
| Cartesia Sonic 3.6 | 5.3% | 440 ms | cloud |
| **ToastTTS + amy-medium** (default voice) | **5.9%** | **97 ms** | laptop CPU |
| ElevenLabs Flash v2.5 | 6.5% | 185 ms | cloud |

Cloud TTFA includes the network round trip; ours doesn't. About 3.5 WER points of ours come from how Whisper writes times ("2.30 pm"). Full table, method and caveats: [research/notebook/06](research/notebook/06-coval-comparison.md).

All findings, methods and decisions: **[research/](research/README.md)**.

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Windows commands shown.

```
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python -m piper.download_voices --download-dir models en_US-amy-medium
.venv\Scripts\python scripts\live.py --play
```

Optional, for the more expressive but slower Kokoro voices ([notebook/10](research/notebook/10-kokoro.md)): put `kokoro-v1.0.onnx` and `voices-v1.0.bin` from the [kokoro-onnx model-files-v1.1 release](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.1) in `models/kokoro/`, then use `--voice kokoro:af_heart`.

`live.py` streams a reply from a simulated LLM (30 tokens/s), plays it live, and prints when each piece was cut plus the TTFA. Try `--tokens-per-second 10`, or `--text samples\no_early_comma.txt`.

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

tts = ToastEngine()                      # amy at 1.2x (the tuned defaults), loaded once
result = tts.say("Well, let me check that for you.")   # plays live
print(result.ttfa_ms, result.pieces)

tts.say(token for token in llm_stream)   # any iterable of text: speaks as it arrives
audio = tts.synthesize("Any text.")      # float32 numpy array at tts.sample_rate
tts.save("Any text.", "out.wav")
```

## Repository map

| Path | Contents |
|---|---|
| [`toast/`](toast/) | The engine: `engine` (the public entry point, `ToastEngine`), `text_normalize`, `stream_chunker` (where to cut), `pacing` (pauses, fades, room tone), `voices` (one interface for Piper/Kitten), `metrics`, `word_check` (Whisper), `naturalness` (UTMOS), `speech_stats` |
| [`scripts/`](scripts/) | Tools: `say.py` (speak any text: typed, argument or piped), `audio_check.py` (do your speakers swallow the first word?), `live.py` (streaming demo), `render.py` (recordings + timing), `bench_voices.py` (all voices), `bench_coval.py` (cloud comparison), `find_voices.py` (multi-speaker scan) |
| [`research/`](research/README.md) | Write-up: lab notebook 01–08, methodology, decision log, original plan |
| [`experiments/`](experiments/README.md) | E01–E07: re-runnable investigations with saved outputs |
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

Pacing, streaming and measurement are working. The open problem is **prosody**: stress, pre-pause shaping and question intonation, which the cloud voices have and a 15M-parameter model doesn't. Next steps (a more expressive model such as Kokoro, distillation, a learned pause model) are in [research/notebook/08](research/notebook/08-open-questions.md).
