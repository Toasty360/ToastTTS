"""E13/E14: Can voice conversion carry a source's delivery into amy's voice?

Step 0 of research/distillation-plan.md. A source speaks 10 sentences (good
delivery); open voice-conversion models re-voice them as amy; we check the
melody survives and the result sounds like amy.
  E13: --source kokoro (default), converters knnvc + openvoice
  E14: --source dailytalk (human conversational speech, CC BY-SA 4.0,
       held-out conversations), converters knnvc + seedvc + seedvc_f0
       (Seed-VC tests the 16 kHz bandwidth problem: 22 / 44.1 kHz output)

Where things run (credit-saving pattern from the author's Gaze-Detection repo):
  laptop (free)  Kokoro source clips, amy reference audio (~10 min of amy
                 reading her own training sentences), amy direct renders
  Modal T4       kNN-VC and OpenVoice-v2 conversion, one job each, hard
                 timeouts, model weights cached in a Volume
  laptop (free)  all evaluation

Never waste a GPU run (lesson from the Gaze project, where a failing final
export lost the computation):
  - every converted clip is validated and written to the Modal Volume and
    committed the moment it exists, before anything else can fail
  - the laptop writes returned audio to disk before evaluating anything
  - `--fetch RUN_ID` re-downloads a finished run from the Volume, no GPU
  - `--smoke` runs the whole chain (incl. saving + fetching) on 1 clip

  .venv\\Scripts\\python -m modal run experiments\\e13_vc_feasibility.py --smoke
  .venv\\Scripts\\python -m modal run experiments\\e13_vc_feasibility.py
  .venv\\Scripts\\python -m modal run experiments\\e13_vc_feasibility.py --fetch 20260925-2130
  .venv\\Scripts\\python -m modal run experiments\\e13_vc_feasibility.py --source dailytalk --converters knnvc,seedvc,seedvc_f0
"""

import io
import time
from pathlib import Path

import modal

app = modal.App("toasttts-vc-feasibility")
cache = modal.Volume.from_name("toasttts-cache", create_if_missing=True)
CACHE = "/cache"

knn_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.5.1", "torchaudio==2.5.1", "numpy<2", "soundfile")
)
openvoice_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .pip_install("torch==2.2.2", "torchaudio==2.2.2", "numpy<2", "librosa==0.10.1", "soundfile", "scipy",
                 "unidecode", "eng_to_ipa", "inflect", "jieba", "pypinyin", "cn2an", "langid", "pydub",
                 "huggingface_hub")
    .run_commands("pip install --no-deps git+https://github.com/myshell-ai/OpenVoice.git")
)
OPENVOICE_REPO = "myshell-ai/OpenVoiceV2"  # the old S3 zip link returns 404 (smoke run 2026-09-25)
# Seed-VC (GPL-3.0 code): 22 kHz speech model, and a 44.1 kHz model that follows the source's
# pitch contour (--f0-condition) shifted into the target's range (--auto-f0-adjust).
seedvc_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .run_commands("git clone --depth 1 https://github.com/Plachtaa/seed-vc /opt/seed-vc",
                  "pip install -r /opt/seed-vc/requirements.txt",
                  # Seed-VC's requirements pin an old protobuf that breaks Modal's own client
                  # inside the container (crash loop in the 2026-09-25 smoke run).
                  "pip install --upgrade 'protobuf>=4.25,<6'")
)


# ---------------------------------------------------------------- remote side

def _wav_bytes(audio, sample_rate):
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _validate(audio, sample_rate, min_seconds=0.3):
    import numpy as np

    audio = np.asarray(audio, dtype="float32").reshape(-1)
    if len(audio) < sample_rate * min_seconds or not np.all(np.isfinite(audio)) or np.abs(audio).max() < 1e-4:
        raise ValueError(f"bad output: {len(audio) / sample_rate:.2f}s, peak {np.abs(audio).max():.5f}")
    return audio


def _save_to_volume(run_id, method, index, wav):
    out = Path(CACHE) / "results" / run_id / method
    out.mkdir(parents=True, exist_ok=True)
    (out / f"clip_{index:02d}.wav").write_bytes(wav)
    cache.commit()  # persisted now: a later failure can't lose this clip


def _write_inputs(files, prefix):
    paths = []
    for i, data in enumerate(files):
        path = f"/tmp/{prefix}_{i:02d}.wav"
        Path(path).write_bytes(data)
        paths.append(path)
    return paths


@app.function(image=knn_image, gpu="T4", timeout=900, volumes={CACHE: cache})
def convert_knnvc(run_id, sources, references):
    import torch

    started = time.time()
    torch.hub.set_dir(f"{CACHE}/torch_hub")  # WavLM + vocoder weights cached across runs
    model = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True,
                           device="cuda")
    matching_set = model.get_matching_set(_write_inputs(references, "ref"))
    outputs = []
    for i, path in enumerate(_write_inputs(sources, "src")):
        audio = _validate(model.match(model.get_features(path), matching_set, topk=4).cpu().numpy(), 16000)
        wav = _wav_bytes(audio, 16000)
        _save_to_volume(run_id, "knnvc", i, wav)
        outputs.append(wav)
    return {"method": "knnvc", "wavs": outputs, "sample_rate": 16000, "gpu_seconds": round(time.time() - started)}


@app.function(image=openvoice_image, gpu="T4", timeout=900, volumes={CACHE: cache})
def convert_openvoice(run_id, sources, references):
    import sys
    import types

    from huggingface_hub import hf_hub_download

    started = time.time()
    ckpt = Path(CACHE) / "openvoice" / "v2"
    if not (ckpt / "converter" / "checkpoint.pth").exists():
        for name in ("converter/config.json", "converter/checkpoint.pth"):
            hf_hub_download(OPENVOICE_REPO, name, local_dir=str(ckpt))
        cache.commit()
    # OpenVoice watermarks its output unless wavmark is missing; training data must not be
    # watermarked, so give it a stub whose model is None (no watermark is added).
    sys.modules["wavmark"] = types.SimpleNamespace(load_model=lambda: types.SimpleNamespace(to=lambda d: None))
    from openvoice.api import ToneColorConverter

    converter = ToneColorConverter(str(ckpt / "converter" / "config.json"), device="cuda")
    converter.load_ckpt(str(ckpt / "converter" / "checkpoint.pth"))
    converter.watermark_model = None
    sample_rate = converter.hps.data.sampling_rate
    source_paths = _write_inputs(sources, "src")
    target_se = converter.extract_se(_write_inputs(references, "ref"))
    source_se = converter.extract_se(source_paths)
    outputs = []
    for i, path in enumerate(source_paths):
        audio = _validate(converter.convert(audio_src_path=path, src_se=source_se, tgt_se=target_se), sample_rate)
        wav = _wav_bytes(audio, sample_rate)
        _save_to_volume(run_id, "openvoice", i, wav)
        outputs.append(wav)
    return {"method": "openvoice", "wavs": outputs, "sample_rate": sample_rate,
            "gpu_seconds": round(time.time() - started)}


@app.function(image=seedvc_image, gpu="T4", timeout=1500, volumes={CACHE: cache})
def convert_seedvc(run_id, sources, references, f0_condition):
    """Runs Seed-VC's own inference.py per clip (loads the model each time: slower, but uses
    the project's tested entry point; fine for a handful of clips)."""
    import os
    import subprocess

    import numpy as np
    import soundfile as sf

    started = time.time()
    method = "seedvc_f0" if f0_condition else "seedvc"
    env = {**os.environ, "HF_HOME": f"{CACHE}/hf"}  # model downloads cached across runs
    # Seed-VC takes one reference clip (1-30 s): use ~25 s of amy.
    ref_audio, ref_sr = sf.read(io.BytesIO(references[0]), dtype="float32")
    ref_path = "/tmp/amy_ref.wav"
    sf.write(ref_path, ref_audio[: int(25 * ref_sr)], ref_sr)
    outputs, sample_rate = [], None
    for i, path in enumerate(_write_inputs(sources, "src")):
        out_dir = f"/tmp/out_{i:02d}"
        flag = "True" if f0_condition else "False"
        subprocess.run(["python", "inference.py", "--source", path, "--target", ref_path, "--output", out_dir,
                        "--diffusion-steps", "30", "--length-adjust", "1.0", "--inference-cfg-rate", "0.7",
                        "--f0-condition", flag, "--auto-f0-adjust", flag, "--semi-tone-shift", "0",
                        "--fp16", "True"], cwd="/opt/seed-vc", env=env, check=True)
        produced = sorted(Path(out_dir).glob("*.wav"))
        audio, sample_rate = sf.read(produced[-1], dtype="float32", always_2d=True)
        audio = _validate(np.asarray(audio).mean(axis=1), sample_rate)
        wav = _wav_bytes(audio, sample_rate)
        _save_to_volume(run_id, method, i, wav)
        outputs.append(wav)
        cache.commit()
    return {"method": method, "wavs": outputs, "sample_rate": sample_rate, "gpu_seconds": round(time.time() - started)}


# ---------------------------------------------------------------- laptop side

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "e13"
METHODS = ("knnvc", "openvoice", "seedvc", "seedvc_f0")
DAILYTALK = ROOT / "models" / "dailytalk" / "data" / "train-00000-of-00009.parquet"
DAILYTALK_FEMALE = 1  # median pitch 212 Hz vs 156 Hz for speaker 0


def is_heldout_conversation(conversation_id):
    """DailyTalk conversations kept out of any training set (10% of them), so
    evaluation clips are never seen in training."""
    return conversation_id % 10 == 0


def _dailytalk_clips(count):
    import pyarrow.parquet as pq

    rows = pq.read_table(DAILYTALK, columns=["conversation_id", "speaker_id", "text", "audio"]).to_pylist()
    rows = [r for r in rows if r["speaker_id"] == DAILYTALK_FEMALE and is_heldout_conversation(r["conversation_id"])
            and len(r["text"].split()) >= 6]
    rows.sort(key=lambda r: (not r["text"].startswith("Well,"), r["conversation_id"]))  # a "Well," clip first
    return [(r["text"], r["audio"]["bytes"]) for r in rows[:count]]


def _test_sentences():
    from toast.splitter import split_into_sentences

    reference = split_into_sentences((ROOT / "samples" / "reference.txt").read_text(encoding="utf-8-sig"))
    word_test = split_into_sentences((ROOT / "samples" / "word_test.txt").read_text(encoding="utf-8-sig"))
    return reference + word_test  # 5 + 5; the first one starts with "Well,"


def _prepare(count, source="kokoro"):
    """Source clips (Kokoro, or human DailyTalk speech), amy reference (~10 min of amy reading her
    own training texts), and amy direct renders of the same sentences."""
    import gzip
    import json

    import numpy as np
    from huggingface_hub import hf_hub_download

    from toast.pacing import trim_silence
    from toast.text_normalize import normalize_for_speech
    from toast.voices import load_voice

    amy = load_voice("amy")
    if source == "kokoro":
        sentences = [normalize_for_speech(s) for s in _test_sentences()[:count]]
        kokoro = load_voice("kokoro:af_heart")
        sources = [_wav_bytes(trim_silence(kokoro.synthesize(s), kokoro.sample_rate), kokoro.sample_rate)
                   for s in sentences]
    else:
        import soundfile as sf

        clips = _dailytalk_clips(count)
        sentences = [text for text, _ in clips]
        sources = []
        for _, data in clips:
            audio, sr = sf.read(io.BytesIO(data), dtype="float32")
            sources.append(_wav_bytes(trim_silence(audio, sr), sr))
    direct = [_wav_bytes(trim_silence(amy.synthesize(normalize_for_speech(s)), amy.sample_rate), amy.sample_rate)
              for s in sentences]

    dataset = hf_hub_download("rhasspy/piper-checkpoints", "en/en_US/amy/medium/dataset.jsonl.gz", repo_type="dataset")
    texts = [json.loads(line)["text"] for line in gzip.open(dataset, "rt", encoding="utf-8")]
    references, chunk, seconds = [], [], 0.0
    for text in texts:  # ~60 s per reference file, ~10 min in total
        audio = trim_silence(amy.synthesize(text), amy.sample_rate)
        chunk += [audio, np.zeros(int(0.3 * amy.sample_rate), dtype=np.float32)]
        seconds += len(audio) / amy.sample_rate + 0.3
        if sum(len(c) for c in chunk) >= 60 * amy.sample_rate:
            references.append(_wav_bytes(np.concatenate(chunk), amy.sample_rate))
            chunk = []
        if seconds >= 600:
            break
    return sentences, sources, direct, references


def _save_local(run_id, name, wavs):
    folder = OUT / run_id / name
    folder.mkdir(parents=True, exist_ok=True)
    for i, wav in enumerate(wavs):
        (folder / f"clip_{i:02d}.wav").write_bytes(wav)
    return folder


def _fetch(run_id):
    """Download a finished run's converted clips from the Volume (no GPU)."""
    for method in METHODS:
        remote = f"results/{run_id}/{method}"
        try:
            entries = sorted(e.path for e in cache.listdir(remote))
        except Exception:
            print(f"  {method}: nothing saved for {run_id}")
            continue
        wavs = [b"".join(cache.read_file(path)) for path in entries]
        print(f"  {method}: fetched {len(wavs)} clips -> {_save_local(run_id, method, wavs)}")


@app.local_entrypoint()
def main(smoke: bool = False, fetch: str = "", source: str = "kokoro", converters: str = "knnvc,openvoice"):
    if fetch:
        _fetch(fetch)
        return
    run_id = time.strftime("%Y%m%d-%H%M") + f"-{source}" + ("-smoke" if smoke else "")
    count = 1 if smoke else 10
    print(f"[{run_id}] preparing inputs on the laptop ({count} {source} sentence(s))...")
    sentences, sources, direct, references = _prepare(count, source)
    _save_local(run_id, f"{source}_source", sources)
    _save_local(run_id, "amy_direct", direct)
    (OUT / run_id / "sentences.txt").write_text("\n".join(sentences), encoding="utf-8")
    print(f"  {len(references)} amy reference files; sending to Modal (T4)...")

    launch = {
        "knnvc": lambda: convert_knnvc.spawn(run_id, sources, references),
        "openvoice": lambda: convert_openvoice.spawn(run_id, sources, references),
        "seedvc": lambda: convert_seedvc.spawn(run_id, sources, references, False),
        "seedvc_f0": lambda: convert_seedvc.spawn(run_id, sources, references, True),
    }
    chosen = [c.strip() for c in converters.split(",") if c.strip()]
    jobs = [launch[c]() for c in chosen]
    for method, job in zip(chosen, jobs):
        try:
            # Don't wait forever: a container that crash-loops at startup never returns.
            result = job.get(timeout=1800)
        except Exception as error:  # one converter failing mustn't lose the other's results
            print(f"  {method}: FAILED on Modal: {error!r}")
            continue
        folder = _save_local(run_id, method, result["wavs"])
        print(f"  {method}: {len(result['wavs'])} clips at {result['sample_rate']} Hz, "
              f"{result['gpu_seconds']} s on GPU -> saved {folder}")
    print(f"\nAll audio is saved in {OUT / run_id} (and on the Modal Volume). Evaluate with:\n"
          f"  .venv\\Scripts\\python experiments\\e13_evaluate.py {run_id}")
