"""Convert the DailyTalk training clips into amy's voice with Seed-VC (f0-conditioned).

E14 chose Seed-VC's pitch-following 44.1 kHz model: the only converter that kept
>= 85% of the human melody while staying bright and sounding like amy
(research/notebook/14). This runs it on the whole training set.

Credit safety (D33):
  - sources and amy's reference are uploaded to the Volume once; containers read them there
  - models load ONCE per container (Seed-VC's inference.py reloads per call; we cache it)
  - every converted clip is written to the Volume immediately, committed every 20 clips
  - resumable: clips already on the Volume are skipped, so a timeout/crash never redoes work
  - --smoke converts 3 clips through the full path first
  - --fetch downloads results without any compute

  .venv\\Scripts\\python -m modal run distill\\convert_modal.py --smoke
  .venv\\Scripts\\python -m modal run distill\\convert_modal.py
  .venv\\Scripts\\python -m modal run distill\\convert_modal.py --fetch
"""

import csv
import io
import time
from pathlib import Path

import modal

app = modal.App("toasttts-convert")
cache = modal.Volume.from_name("toasttts-cache", create_if_missing=True)
CACHE = "/cache"
RUN = "dailytalk_f1_seedvc_f0"  # one fixed output folder, so re-runs resume instead of duplicating
SHARDS = 4

seedvc_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .run_commands("git clone --depth 1 https://github.com/Plachtaa/seed-vc /opt/seed-vc",
                  "pip install -r /opt/seed-vc/requirements.txt",
                  "pip install --upgrade 'protobuf>=4.25,<6'")  # see e13: old pin breaks Modal
)


@app.function(image=seedvc_image, gpu="T4", timeout=4 * 3600, volumes={CACHE: cache})
def convert_shard(ids):
    import argparse
    import os
    import sys

    import numpy as np
    import soundfile as sf

    os.environ["HF_HOME"] = f"{CACHE}/hf"
    os.chdir("/opt/seed-vc")
    sys.path.insert(0, "/opt/seed-vc")
    sys.argv = ["inference.py"]
    import inference  # Seed-VC's own script

    loaded = inference.load_models  # load once, reuse for every clip
    cached = {}
    inference.load_models = lambda args: cached.setdefault("models", loaded(args))

    out_dir = Path(CACHE) / "results" / RUN
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = f"{CACHE}/sources/amy_reference.wav"
    done, skipped, failed = [], 0, []
    started = time.time()
    for n, clip_id in enumerate(ids, 1):
        target = out_dir / f"{clip_id}.wav"
        if target.exists():
            skipped += 1
            continue
        args = argparse.Namespace(
            source=f"{CACHE}/sources/dailytalk_f1/{clip_id}.wav", target=reference, output="/tmp/vc",
            diffusion_steps=30, length_adjust=1.0, inference_cfg_rate=0.7, f0_condition=True,
            auto_f0_adjust=True, semi_tone_shift=0, checkpoint=None, config=None, fp16=True)
        try:
            for old in Path("/tmp/vc").glob("*.wav") if Path("/tmp/vc").exists() else []:
                old.unlink()
            inference.main(args)
            audio, sr = sf.read(next(Path("/tmp/vc").glob("*.wav")), dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            fade = int(0.015 * sr)  # Seed-VC output starts at sample 0; a click read as "as" (E14)
            audio[:fade] *= np.linspace(0.0, 1.0, fade)
            if len(audio) < sr * 0.3 or not np.all(np.isfinite(audio)) or np.abs(audio).max() < 1e-4:
                raise ValueError("bad output")
            sf.write(target, audio, sr, subtype="PCM_16")
            done.append(clip_id)
        except Exception as error:
            failed.append((clip_id, repr(error)))
        if n % 20 == 0:
            cache.commit()
            print(f"[shard] {n}/{len(ids)} ({time.time() - started:.0f} s)", flush=True)
    cache.commit()
    return {"done": len(done), "skipped": skipped, "failed": failed, "seconds": round(time.time() - started)}


# ------------------------------------------------------------------ laptop side

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "distill" / "data"


def _clip_ids():
    with (DATA / "sources.csv").open(encoding="utf-8") as f:
        return [row["id"] for row in csv.DictReader(f, delimiter="|")]


def _make_reference():
    """~25 s of amy reading her own original training sentences (Seed-VC takes one 1-30 s clip)."""
    import gzip
    import json

    import numpy as np
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    from toast.pacing import trim_silence
    from toast.voices import load_voice

    path = DATA / "amy_reference.wav"
    if path.exists():
        return path
    amy = load_voice("amy")
    dataset = hf_hub_download("rhasspy/piper-checkpoints", "en/en_US/amy/medium/dataset.jsonl.gz", repo_type="dataset")
    parts = []
    for line in gzip.open(dataset, "rt", encoding="utf-8"):
        parts += [trim_silence(amy.synthesize(json.loads(line)["text"]), amy.sample_rate),
                  np.zeros(int(0.3 * amy.sample_rate), dtype=np.float32)]
        if sum(len(p) for p in parts) > 25 * amy.sample_rate:
            break
    sf.write(path, np.concatenate(parts)[: 25 * amy.sample_rate], amy.sample_rate)
    return path


def _upload():
    reference = _make_reference()
    with cache.batch_upload(force=True) as batch:
        batch.put_file(str(reference), "/sources/amy_reference.wav")
        batch.put_directory(str(DATA / "sources"), "/sources/dailytalk_f1")
    print(f"uploaded {len(list((DATA / 'sources').glob('*.wav')))} source clips + amy reference")


def _fetch():
    out = DATA / "converted" / RUN
    out.mkdir(parents=True, exist_ok=True)
    have = {p.name for p in out.glob("*.wav")}
    got = 0
    for entry in cache.listdir(f"results/{RUN}"):
        name = Path(entry.path).name
        if name not in have:
            (out / name).write_bytes(b"".join(cache.read_file(entry.path)))
            got += 1
    print(f"fetched {got} new clips -> {out} ({len(list(out.glob('*.wav')))} total)")


@app.local_entrypoint()
def main(smoke: bool = False, fetch: bool = False):
    if fetch:
        _fetch()
        return
    _upload()
    ids = _clip_ids()[:3] if smoke else _clip_ids()
    shards = [ids[i::SHARDS] for i in range(SHARDS)] if not smoke else [ids]
    print(f"converting {len(ids)} clips in {len(shards)} container(s)...")
    for result in convert_shard.map(shards):
        print(f"  shard: {result['done']} converted, {result['skipped']} already done, "
              f"{len(result['failed'])} failed, {result['seconds']} s", flush=True)
        for clip_id, error in result["failed"][:5]:
            print(f"    failed {clip_id}: {error}")
    _fetch()  # everything is on the Volume already; this just brings it to the laptop
