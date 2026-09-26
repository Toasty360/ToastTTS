"""e18: transcribe LibriVox chapters on a Modal T4 GPU with openai-whisper.

Same transcriber as e16 (small.en, word_timestamps) so labels stay comparable.
Reads mp3s from the e18-audio volume, writes word JSONs to e18-words.
Run: modal run modal_transcribe.py
"""
import modal

app = modal.App("e18-transcribe")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("openai-whisper", "numpy")
)

audio_vol = modal.Volume.from_name("e18-audio", create_if_missing=True)
words_vol = modal.Volume.from_name("e18-words", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/vol_audio": audio_vol, "/vol_words": words_vol},
    timeout=3600,
)
def transcribe_all():
    import glob
    import json
    import os
    import time

    import whisper

    print("loading whisper small.en on cuda...", flush=True)
    model = whisper.load_model("small.en", device="cuda")
    files = sorted(glob.glob("/vol_audio/**/*.mp3", recursive=True))
    print(f"{len(files)} files", flush=True)
    for i, fp in enumerate(files):
        book = os.path.basename(os.path.dirname(fp))
        stem = os.path.splitext(os.path.basename(fp))[0]
        out = f"/vol_words/{book}_{stem}.json"
        if os.path.exists(out):
            print(f"[{i+1}/{len(files)}] skip {book}/{stem} (exists)", flush=True)
            continue
        t0 = time.time()
        try:
            result = model.transcribe(fp, word_timestamps=True,
                                      language="en", verbose=False)
        except Exception as e:
            print(f"[{i+1}/{len(files)}] ERROR {fp}: {e}", flush=True)
            continue
        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                txt = w["word"].strip()
                if txt:
                    words.append({"w": txt, "start": round(w["start"], 3),
                                  "end": round(w["end"], 3)})
        json.dump({"file": fp, "words": words}, open(out, "w"))
        dt = time.time() - t0
        dur = words[-1]["end"] if words else 0.0
        rtf = dt / max(dur, 0.01)
        print(f"[{i+1}/{len(files)}] {book}/{stem} words={len(words)} "
              f"audio={dur:.1f}s wall={dt:.1f}s rtf={rtf:.2f}x", flush=True)
    words_vol.commit()
    print("DONE", flush=True)


@app.local_entrypoint()
def main():
    transcribe_all.remote()
