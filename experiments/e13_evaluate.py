"""E13/E14 evaluation (laptop only, no GPU): did voice conversion keep the
source's delivery, does it sound like amy, and is the audio clean and bright?

Reads out/e13/<run_id>/ written by e13_vc_feasibility.py. Pass criteria
from research/distillation-plan.md, Step 0 / 0b:
  melody     core pitch range (10-90th pct) at least 85% of the source's
             (percentile-trimmed: E13 showed 5-95% range is inflated by outliers)
  identity   closer to amy than to the source, and >= 80% of amy's own consistency
  words      no more wrong words than amy's own reading of the same sentences
             (was "0" in the first E13 run, where amy had 0; DailyTalk text trips amy too)
  clean      pitch-jump rate no worse than 2x amy's
  bandwidth  energy above 7 kHz at least 50% of amy's (else the student may sound dull)
  (listening decides the rest)

  .venv\\Scripts\\python experiments\\e13_evaluate.py RUN_ID [listening-file prefix]
"""

import sys
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e12_well import describe_word  # noqa: E402

from toast.naturalness import naturalness  # noqa: E402
from toast.speech_stats import timed_words  # noqa: E402
from toast.word_check import transcribe, word_errors  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_encoder = None


def speaker_embedding(audio, sr):
    global _encoder
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    if _encoder is None:
        _encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"},
                                                  savedir=str(ROOT / "models" / "ecapa"), local_strategy=LocalStrategy.COPY)
    t = np.linspace(0, len(audio) / sr, int(len(audio) * 16000 / sr), endpoint=False)
    audio16 = np.interp(t, np.arange(len(audio)) / sr, audio).astype(np.float32)
    e = _encoder.encode_batch(torch.from_numpy(audio16).unsqueeze(0)).squeeze().numpy()
    return e / np.linalg.norm(e)


def load(folder):
    clips = []
    for path in sorted(folder.glob("clip_*.wav")):
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        clips.append((audio.mean(axis=1), sr))
    return clips


def signal_stats(clips):
    """Core pitch range, pitch-jump rate, share of energy above 7 kHz, brightness."""
    semitones, jumps, voiced_frames, high, total, centroids = [], 0, 0, 0.0, 0.0, []
    for audio, sr in clips:
        hz = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr).to_pitch_ac(
            time_step=0.01, pitch_floor=75, pitch_ceiling=500).selected_array["frequency"]
        voiced = hz > 0
        st = 12 * np.log2(np.where(voiced, hz, 1.0) / 100.0)
        both = voiced[1:] & voiced[:-1]
        jumps += int(np.sum(np.abs(np.diff(st))[both] > 6))
        voiced_frames += int(voiced.sum())
        semitones += list(st[voiced])
        spectrum = np.abs(np.fft.rfft(audio)) ** 2
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)
        high += spectrum[freqs > 7000].sum()
        total += spectrum.sum()
        centroids.append(np.sum(freqs * spectrum) / np.sum(spectrum))
    semitones = np.array(semitones)
    return {"core_range": float(np.percentile(semitones, 90) - np.percentile(semitones, 10)),
            "jumps": 100 * jumps / max(voiced_frames, 1), "high": 100 * high / total,
            "brightness": float(np.mean(centroids))}


def main(run_id, prefix="60_vc"):
    run = ROOT / "out" / "e13" / run_id
    sentences = (run / "sentences.txt").read_text(encoding="utf-8").splitlines()
    names = sorted(p.name for p in run.iterdir() if p.is_dir())
    source = next(n for n in names if n.endswith("_source"))
    converters = [n for n in names if n not in (source, "amy_direct")]
    data = {name: load(run / name) for name in [source, "amy_direct", *converters]}

    amy_embs = [speaker_embedding(a, sr) for a, sr in data["amy_direct"]]
    src_embs = [speaker_embedding(a, sr) for a, sr in data[source]]
    amy_centre, src_centre = np.mean(amy_embs, axis=0), np.mean(src_embs, axis=0)
    amy_self = float(np.mean([e @ (np.sum(amy_embs, axis=0) - e) / max(len(amy_embs) - 1, 1) for e in amy_embs])) \
        if len(amy_embs) > 1 else 1.0

    print(f"Run {run_id}: {len(sentences)} sentence(s), source = {source}\n")
    print(f"{'condition':<16}{'core st':>8}{'jumps%':>7}{'>7kHz%':>8}{'bright Hz':>10}{'sim amy':>8}{'sim src':>8}"
          f"{'wrong':>6}{'natural':>8}")
    results = {}
    for name, clips in data.items():
        sr = clips[0][1]
        gap = np.zeros(int(0.4 * sr), dtype=np.float32)
        sf.write(ROOT / "out" / "listen" / f"{prefix}_{name}.wav",
                 np.concatenate([x for a, _ in clips for x in (a, gap)]), sr)
        stats = signal_stats(clips)
        embs = [speaker_embedding(a, s) for a, s in clips]
        stats["sim_amy"] = float(np.mean([e @ amy_centre / np.linalg.norm(amy_centre) for e in embs]))
        stats["sim_src"] = float(np.mean([e @ src_centre / np.linalg.norm(src_centre) for e in embs]))
        stats["wrong"] = sum(word_errors(s, transcribe(a, sr))[0] for s, (a, _) in zip(sentences, clips))
        stats["natural"] = float(np.mean([naturalness(a, sr) for a, _ in clips]))
        results[name] = stats
        print(f"{name:<16}{stats['core_range']:>8.2f}{stats['jumps']:>7.2f}{stats['high']:>8.3f}"
              f"{stats['brightness']:>10.0f}{stats['sim_amy']:>8.2f}{stats['sim_src']:>8.2f}"
              f"{stats['wrong']:>6}{stats['natural']:>8.2f}")

    print(f"\nFirst sentence: {sentences[0]!r}")
    for name, clips in data.items():
        word, start, end = timed_words(*clips[0])[0]
        try:
            shape = describe_word(clips[0][0], clips[0][1], start, end)
        except Exception as error:  # Whisper can return a near-zero-length first word
            shape = f"could not analyse ({type(error).__name__}; word boundary {start:.2f}-{end:.2f} s)"
        print(f"  {name:<16}{word!r:<9}{shape}")

    src, amy = results[source], results["amy_direct"]
    print(f"\namy's own consistency: {amy_self:.2f}")
    print("\nVerdict per converter:")
    for name in converters:
        r = results[name]
        checks = {
            f"melody >= 85% of source ({0.85 * src['core_range']:.1f} st)": r["core_range"] >= 0.85 * src["core_range"],
            "closer to amy than source": r["sim_amy"] > r["sim_src"],
            f"identity >= 80% of amy's ({0.8 * amy_self:.2f})": r["sim_amy"] >= 0.8 * amy_self,
            # amy herself mis-says some DailyTalk words (names, casual speech), so the fair bar
            # is "no worse than amy's own reading of the same sentences".
            f"wrong words <= amy's ({amy['wrong']})": r["wrong"] <= amy["wrong"],
            "clean (jumps <= 2x amy)": r["jumps"] <= 2 * max(amy["jumps"], 0.1),
            "bright (>7kHz >= 50% of amy)": r["high"] >= 0.5 * amy["high"],
        }
        verdict = "PASS" if all(checks.values()) else "FAIL"
        print(f"  {name}: {verdict}   " + "   ".join(f"{k}: {'yes' if v else 'NO'}" for k, v in checks.items()))
    print(f"\nListen: out/listen/{prefix}_*.wav")


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
