"""Fine-tune amy-medium on converted speech, on Modal, without ever losing work.

What runs where:
  laptop        builds distill/data/train/<dataset>/ (wavs + metadata.csv) and uploads it
  Modal GPU     piper1-gpl training from amy's checkpoint (train)
  Modal CPU     ONNX export + a test synthesis (export), no GPU cost
  laptop        evaluation (distill/evaluate_student.py)

Safeguards (research/decisions.md D33; the author once lost a run to a failing
final export):
  - smoke mode: 10 minutes of training (small batches), then export, then fetch.
    The whole chain is proven for about $0.15 before a real run.
  - metadata.csv is "id|text" with NO header row: piper reads every line as a clip.
  - checkpoints are written straight into the Modal Volume, which is committed
    every 5 minutes while training runs and once more at the end, even on error.
  - the training time is capped (piper's default is to train forever).
  - export is a separate CPU step that only reads checkpoints; if it fails, the
    checkpoints are untouched and export can be re-run alone (--export-only).
  - every exported ONNX is loaded with onnxruntime and made to speak before
    it counts; the file is saved to the Volume before it's returned.
  - --fetch RUN pulls a finished run's exports from the Volume, no compute.
  - torch is pinned (2.5.1) to the TorchScript ONNX exporter piper was written for.

  .venv\\Scripts\\python -m modal run distill\\train_modal.py --dataset DATASET --smoke
  .venv\\Scripts\\python -m modal run distill\\train_modal.py --dataset DATASET --hours 2      # stage A
  .venv\\Scripts\\python -m modal run distill\\train_modal.py --dataset DATASET --hours 2 --resume RUN_ID  # continue
  .venv\\Scripts\\python -m modal run distill\\train_modal.py --export-only RUN_ID
  .venv\\Scripts\\python -m modal run distill\\train_modal.py --fetch RUN_ID
"""

import json
import time
from pathlib import Path

import modal

PIPER_COMMIT = "main"  # recorded in each run's manifest (resolved hash) for reproducibility
AMY_CKPT = "en/en_US/amy/medium/epoch=6679-step=1554200.ckpt"
AMY_CONFIG = "en/en_US/amy/medium/config.json"

app = modal.App("toasttts-distill")
volume = modal.Volume.from_name("toasttts-train", create_if_missing=True)
VOL = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake", "ninja-build")
    .pip_install("torch==2.5.1", "torchaudio==2.5.1", "huggingface_hub", "onnxruntime",
                 # needed by piper's own `setup.py build_ext` (only present in pip's isolated build otherwise)
                 "scikit-build", "cmake", "ninja", "cython")
    .run_commands(
        f"git clone https://github.com/OHF-Voice/piper1-gpl /opt/piper && cd /opt/piper && git checkout {PIPER_COMMIT}",
        "cd /opt/piper && pip install -e '.[train]'",
        "cd /opt/piper && ./build_monotonic_align.sh && python setup.py build_ext --inplace",
        # torch stays pinned even if piper's extras pulled something newer
        "pip install torch==2.5.1 torchaudio==2.5.1",
    )
)


def _commit_every(seconds, stop):
    while not stop.wait(seconds):
        try:
            volume.commit()
        except Exception as error:  # never let a commit hiccup kill training
            print(f"[commit] {error!r}", flush=True)


@app.function(image=image, gpu="L4", timeout=8 * 3600, volumes={VOL: volume})
def train(run_id, dataset, hours, smoke, resume_from="", freeze=""):
    import subprocess
    import threading

    from huggingface_hub import hf_hub_download

    run = Path(VOL) / "runs" / run_id
    run.mkdir(parents=True, exist_ok=True)
    data = Path(VOL) / "datasets" / dataset
    volume.reload()
    if resume_from:  # staged training: continue a previous stage from its last checkpoint
        previous = sorted((Path(VOL) / "runs" / resume_from).rglob("last.ckpt"))
        if not previous:
            raise SystemExit(f"no last.ckpt in run {resume_from}")
        return _train(run, data, str(previous[-1]), run_id, dataset, hours, smoke,
                      f"runs/{resume_from}/{previous[-1].relative_to(Path(VOL) / 'runs' / resume_from)}", freeze)
    original = hf_hub_download("rhasspy/piper-checkpoints", AMY_CKPT, repo_type="dataset",
                               cache_dir=f"{VOL}/hf")
    # Newer Lightning CLIs read the hyper-parameters stored in a --ckpt_path checkpoint and
    # pass them as options; amy's 2023 checkpoint carries old ones (e.g. sample_bytes) the
    # current trainer rejects. Keep weights + training state, drop only the stored settings.
    ckpt = Path(VOL) / "checkpoints" / "amy_medium_clean.ckpt"
    if not ckpt.exists():
        import torch

        state = torch.load(original, map_location="cpu", weights_only=False)  # trusted: official rhasspy
        dropped = sorted(state.pop("hyper_parameters", {}).keys())
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, ckpt)
        volume.commit()
        print(f"cleaned amy checkpoint (dropped stored settings: {dropped})", flush=True)
    return _train(run, data, str(ckpt), run_id, dataset, hours, smoke, AMY_CKPT, freeze)


def _train(run, data, ckpt, run_id, dataset, hours, smoke, start_label, freeze=""):
    import subprocess
    import threading

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd="/opt/piper", capture_output=True, text=True).stdout.strip()
    manifest = {"run_id": run_id, "dataset": dataset, "hours": hours, "smoke": smoke, "piper_commit": commit,
                "start_checkpoint": start_label, "freeze": freeze or "nothing",
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))
    volume.commit()

    limit = "00:00:10:00" if smoke else f"00:{int(hours):02d}:{int(hours % 1 * 60):02d}:00"
    # amy's 2023 checkpoint (official rhasspy/piper-checkpoints, trusted) stores Python objects
    # that torch's strict weights_only loader refuses; load it in full mode for this trainer only.
    launcher = Path("/tmp/launch_piper_train.py")
    launcher.write_text(
        "import sys, torch\n"
        "_load = torch.load\n"
        "torch.load = lambda *a, **k: _load(*a, **{**k, 'weights_only': False})\n"
        + (
            # Freeze the decoder (the part that renders the sound wave, i.e. largely the voice
            # itself) so the student keeps amy's voice and learns delivery in the rest. With
            # AdamW, parameters without gradients are skipped (no update, no weight decay).
            "from piper.train.vits.lightning import VitsModel\n"
            "_init = VitsModel.__init__\n"
            "def _frozen_init(self, *a, **k):\n"
            "    _init(self, *a, **k)\n"
            "    n = 0\n"
            "    for p in self.model_g.dec.parameters():\n"
            "        p.requires_grad = False\n"
            "        n += p.numel()\n"
            "    print(f'[freeze] decoder frozen: {n:,} parameters', flush=True)\n"
            "VitsModel.__init__ = _frozen_init\n" if freeze == "decoder" else "")
        + "from piper.train.__main__ import main\n"
        "sys.argv[0] = 'piper.train'\n"
        "main()\n")
    command = [
        "python", str(launcher), "fit",
        "--data.voice_name", "amy_distill",
        "--data.csv_path", str(data / "metadata.csv"),
        "--data.audio_dir", str(data / "wavs"),
        "--model.sample_rate", "22050",
        "--data.espeak_voice", "en-us",
        "--data.cache_dir", "/tmp/piper_cache",
        "--data.config_path", str(run / "config.json"),
        "--data.batch_size", "16" if smoke else "32",
        "--ckpt_path", ckpt,
        "--trainer.default_root_dir", str(run),
        "--trainer.max_time", limit,
        # validation (incl. a UTMOS score on generated audio) every epoch is a real share of
        # GPU time on ~900 clips; every 5 epochs still feeds the best-checkpoint selection
        "--trainer.check_val_every_n_epoch", "1" if smoke else "5",
    ]
    print(" ".join(command), flush=True)

    stop = threading.Event()
    committer = threading.Thread(target=_commit_every, args=(300, stop), daemon=True)
    committer.start()
    started = time.time()
    try:
        result = subprocess.run(command, cwd="/opt/piper")
        status = "finished" if result.returncode == 0 else f"exit code {result.returncode}"
    finally:
        stop.set()
        manifest.update({"training_seconds": round(time.time() - started), "ended": time.strftime("%Y-%m-%d %H:%M:%S")})
        (run / "manifest.json").write_text(json.dumps(manifest, indent=2))
        volume.commit()  # checkpoints are safe whatever happened above
    checkpoints = sorted(str(p.relative_to(run)) for p in run.rglob("*.ckpt"))
    return {"status": status, "checkpoints": checkpoints, "training_seconds": manifest["training_seconds"]}


@app.function(image=image, cpu=4.0, memory=8192, timeout=1800, volumes={VOL: volume})
def export(run_id, which=("last",)):
    """Export checkpoints to ONNX on CPU, verify each speaks, save to the Volume."""
    import subprocess

    import numpy as np
    import onnxruntime

    volume.reload()
    run = Path(VOL) / "runs" / run_id
    checkpoints = sorted(run.rglob("*.ckpt"))
    chosen = [c for c in checkpoints if any(w in c.name for w in which)] or checkpoints[-1:]
    exported = {}
    for checkpoint in chosen:
        name = checkpoint.stem.replace("=", "_")
        onnx_path = run / "onnx" / f"{name}.onnx"
        try:
            # same full-mode loading as training (our own checkpoints, from our own run)
            subprocess.run(["python", "-c",
                            "import sys, torch; _l = torch.load; "
                            "torch.load = lambda *a, **k: _l(*a, **{**k, 'weights_only': False}); "
                            "from piper.train.export_onnx import main; "
                            f"sys.argv = ['export_onnx', '--checkpoint', {str(checkpoint)!r}, "
                            f"'--output-file', {str(onnx_path)!r}]; main()"],
                           cwd="/opt/piper", check=True)
            # Verify: the file must load and produce sound for a real phoneme sequence.
            session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            id_map = json.loads((run / "config.json").read_text())["phoneme_id_map"]
            sequence = [id_map["^"][0]]
            for phoneme in "həlˈoʊ wˈɜːld":  # "hello world", as Piper's espeak phonemes
                sequence += [id_map[phoneme][0], id_map["_"][0]]
            ids = np.array([sequence + [id_map["$"][0]]], dtype=np.int64)
            audio = session.run(None, {"input": ids, "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
                                       "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32)})[0].squeeze()
            if audio.size < 1000 or not np.all(np.isfinite(audio)):
                raise ValueError(f"exported model produced bad audio ({audio.size} samples)")
            volume.commit()
            exported[name] = {"onnx": onnx_path.read_bytes(), "config": (run / "config.json").read_bytes()}
            print(f"[export] {name}: ok, {onnx_path.stat().st_size / 1e6:.1f} MB", flush=True)
        except Exception as error:  # one bad export must not lose the others
            print(f"[export] {name}: FAILED {error!r}", flush=True)
    return exported


@app.function(image=image, cpu=2.0, memory=8192, timeout=900, volumes={VOL: volume})
def compare_decoder(run_id):
    """Proof for a frozen-decoder run: its decoder weights must equal amy's exactly."""
    import torch

    volume.reload()
    last = sorted((Path(VOL) / "runs" / run_id).rglob("last.ckpt"))[-1]
    amy = torch.load(f"{VOL}/checkpoints/amy_medium_clean.ckpt", map_location="cpu", weights_only=False)["state_dict"]
    run = torch.load(last, map_location="cpu", weights_only=False)["state_dict"]
    report = {}
    for part in ("model_g.dec.", "model_g.enc_p.", "model_g.dp.", "model_g.flow."):
        keys = [k for k in amy if k.startswith(part) and k in run]
        diff = max((run[k].float() - amy[k].float()).abs().max().item() for k in keys) if keys else None
        report[part.rstrip(".")] = {"tensors": len(keys), "max_abs_change": diff}
    return report


# ------------------------------------------------------------------ laptop side

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def _upload(dataset):
    local = ROOT / "distill" / "data" / "train" / dataset
    if not (local / "metadata.csv").exists():
        raise SystemExit(f"{local}/metadata.csv not found; build the dataset first")
    with volume.batch_upload(force=True) as batch:
        batch.put_directory(str(local), f"/datasets/{dataset}")
    print(f"uploaded {dataset}: {len(list((local / 'wavs').glob('*.wav')))} clips")


def _save_exports(run_id, exported, tag=""):
    for name, files in exported.items():
        base = MODELS / f"en_US-amy_distill_{run_id}{tag}_{name}-medium"
        Path(f"{base}.onnx").write_bytes(files["onnx"])
        config = json.loads(files["config"])
        Path(f"{base}.onnx.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"saved {base.name}.onnx (+ .json) -> voice name: piper:{base.name}")


@app.local_entrypoint()
def main(dataset: str = "", smoke: bool = False, hours: float = 2.0, export_only: str = "", fetch: str = "",
         resume: str = "", freeze: str = "", verify_freeze: str = "", which: str = "last,val_mos,val_mel",
         tag: str = "", upload_only: bool = False):
    if upload_only:
        if not dataset:
            raise SystemExit("--dataset is required with --upload-only")
        _upload(dataset)
        return
    if verify_freeze:
        for part, info in compare_decoder.remote(verify_freeze).items():
            print(f"{part:<16} {info['tensors']:>4} tensors, max change vs amy: {info['max_abs_change']}")
        return
    if fetch or export_only:
        run_id = fetch or export_only
        _save_exports(run_id, export.remote(run_id, tuple(which.split(","))), tag)
        return
    if not dataset:
        raise SystemExit("--dataset is required")
    if not smoke and hours > 7.5:
        raise SystemExit("refusing more than 7.5 h in one run (budget, research/distillation-plan.md)")
    run_id = (time.strftime("%Y%m%d-%H%M") + ("-smoke" if smoke else "") + (f"-from-{resume}" if resume else "")
              + (f"-freeze-{freeze}" if freeze else ""))
    if not resume:
        _upload(dataset)
    print(f"[{run_id}] training on Modal ({'smoke' if smoke else f'{hours} h cap'})...")
    result = train.remote(run_id, dataset, hours, smoke, resume, freeze)
    print(f"[{run_id}] training {result['status']} after {result['training_seconds']} s; "
          f"{len(result['checkpoints'])} checkpoints on the Volume")
    _save_exports(run_id, export.remote(run_id, ("last",) if smoke else ("last", "val_mos", "val_mel")))
