# 09: First test on real input: `say.py`

*2026-09-25 · Code: `toast/engine.py` (`ToastEngine`), `scripts/say.py`, `toast/player.py`, `toast/text_normalize.py` · Tests: `tests/test_engine.py`, `tests/test_text_normalize.py`*

## Setup

A public entry point was added so the engine can be tried on any text, not just the prepared samples ([D19](../decisions.md)):
- `ToastEngine`: load once, then `say()`, `synthesize()`, `save()`, `stream()`, for a string or any stream of text;
- `scripts/say.py`: typed, argument or piped input.

## What the first live run showed

The author ran `say.py "Order #4521 ships at 3:15 PM."` with playback:

```
TTFA 273 ms | 5.4 s of audio made in 6.13 s
pieces: Order #4521 ships at | 3:15 PM.
```

Four problems, investigated one by one:

| Problem | Cause (measured) | Fix |
|---|---|---|
| TTFA 273 ms (vs ~40–120 ms without playback) | The sound device was opened on every call (64 ms the first time, ~10 ms after), plus first-use overhead. The rest was a long first piece (118 ms without playback) | `LivePlayer` opens **once** and stays running; `ToastEngine(open_audio=True)` opens it at startup |
| Sound heard later than TTFA suggests | The device ran at the driver's default "high" latency | `latency="low"` |
| "made in 6.13 s" for 5.4 s of audio | It measured until *playback* finished, not generation | Record the time the last chunk was produced (now 0.16 s) |
| "ships at \| 3:15 PM" | The quick first-piece cut avoided numbers but not small leaning words | The flash cut never lands after words like *at, to, the, your, of* (`LEANING_WORDS`) |
| "#4521" read as "hash…" (also in [06](06-coval-comparison.md)) | The pronunciation step spells the symbol | `text_normalize.py`: `#4521` → "number 4521", `#ORD` → "ORD" |

After the fixes, without playback, Whisper heard "Order number 4521 ships at 3:15 PM." with 0 wrong words.

**Trade-off:** that sentence now has no acceptable early cut ("at" leans on the time, and "3:15 PM" stays together), so it's spoken as one piece: **159 ms** TTFA instead of 118 ms with the bad cut. Better phrasing was chosen over 40 ms. Longer replies still get an early cut.

## Second run: the first words are swallowed

After those fixes the author reported: `TTFA 163 ms | 4.8 s of audio made in 0.16 s`, but **the first words weren't heard**. They had seen the same thing in another project (InterviewAgent), which already adds 80 ms of silence before the first word "so the endpoint is already running". So 80 ms of *silence* isn't enough on this machine.

**Probable cause (not yet confirmed by ear):** the output device is a monitor's speakers ("Smart M70F", Samsung Smart Monitor M7, over HDMI/DisplayPort). Monitor and TV speakers commonly mute when they receive pure digital silence and take hundreds of ms to unmute once sound arrives. Our player made this worse in two ways: it sent exact zeros when idle, and it opened the device right before speaking.

**Fix ([D23](../decisions.md)):**
- When idle, `LivePlayer` plays the same -60 dB room tone that runs under our speech, never pure silence.
- At startup, the device runs for 0.6 s (`DEVICE_WAKE_S`) before the first word. This happens once per program run, not per utterance, and isn't counted in TTFA.
- `render.py` and E01 (which used `sd.play`) now use the same `toast.player.play()`.

**Confirmed (2026-09-25):** the idle room tone plus 0.6 s wake did *not* fix it on the monitor ("still not hearing 'Order #'"). Switching Windows output to the **laptop speakers played every word**, and the saved audio transcribes completely ("Order number 4521 ships at 3:15 PM."). So the loss happens in the monitor's speakers, not in ToastTTS. Two settings were added to find what the monitor needs, `say.py --wake SECONDS` and `--idle-db DB`. **Shelved** at the author's request: laptop speakers or headphones work, and per-device tuning can wait. Worth revisiting before anyone else uses ToastTTS on monitor/TV speakers.

**Diagnosis tool:** `scripts/audio_check.py` plays three beeps three ways: immediately, after 1 s of digital silence, and after 0.6 s of room tone. It tells apart "needs time after opening" from "mutes on digital silence".

Also measured: the device's "low" latency is still **91 ms** of output buffering with Windows' default audio path (MME). WASAPI may allow less; not yet tried.

## Note on what TTFA measures here

`ToastEngine` TTFA runs from the call to the first audio chunk being ready. It doesn't include the sound card's own output buffer (now "low" latency, typically tens of ms), which adds to what a listener perceives. Coval-style TTFA ([06](06-coval-comparison.md)) doesn't include it either.

## Not yet measured

- The effect of the `#` rule and the leaning-word rule on the Coval WER and TTFA. It can be re-run free with local Whisper large-v2 (`scripts/bench_coval.py`, about 40 min); compare against `benchmarks/coval_tts_v1.csv`, which used the same local scorer.
- Playback latency end to end (from the call to sound leaving the speaker) needs a loopback recording.

→ Back to [08: open questions](08-open-questions.md)
