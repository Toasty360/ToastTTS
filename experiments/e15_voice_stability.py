"""E15: Is the voice "shaking / breaking"? Standard voice-stability measures.

Listening note: the author heard the converted clips (61_dt_seedvc_f0) and the run C
student at 1 h as "shaking or breaking". E14's pitch-jump check (> 6 st in 10 ms)
only catches large jumps, not small fast wobble, so this uses the standard
clinical voice measures from Praat:
  jitter (local)   cycle-to-cycle variation of pitch period, %   (wobble in pitch)
  shimmer (local)  cycle-to-cycle variation of amplitude, %      (wobble in loudness)
  HNR              harmonics-to-noise ratio, dB                  (clean vs rough/breathy)
Rough reference values for healthy natural voices (MDVP norms): jitter < ~1.0%,
shimmer < ~3.8%, HNR > ~20 dB. Compare conditions against each other and against
the human original, not only against the norms.

Same 10 held-out DailyTalk sentences for every condition (from the E14 run folder);
students are synthesized on them.

  .venv\\Scripts\\python experiments\\e15_voice_stability.py 20260925-2232-dailytalk [student voice ...]
"""

import sys
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf
from parselmouth.praat import call

from toast.pacing import trim_silence
from toast.text_normalize import normalize_for_speech
from toast.voices import load_voice

ROOT = Path(__file__).resolve().parents[1]


def stability(audio, sr):
    sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)
    points = call(sound, "To PointProcess (periodic, cc)", 75, 500)
    jitter = call(points, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = call([sound, points], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    return 100 * jitter, 100 * shimmer, hnr


def summarise(label, clips):
    values = np.array([stability(a, sr) for a, sr in clips])
    j, s, h = np.nanmedian(values, axis=0)
    print(f"{label:<44}{j:>9.2f}{s:>10.2f}{h:>9.1f}")


def main(run_id, students):
    run = ROOT / "out" / "e13" / run_id
    sentences = (run / "sentences.txt").read_text(encoding="utf-8").splitlines()
    print(f"{len(sentences)} held-out DailyTalk sentences; median over clips\n")
    print(f"{'condition':<44}{'jitter %':>9}{'shimmer %':>10}{'HNR dB':>9}")
    for name in sorted(p.name for p in run.iterdir() if p.is_dir()):
        clips = []
        for path in sorted((run / name).glob("clip_*.wav")):
            audio, sr = sf.read(path, dtype="float32", always_2d=True)
            clips.append((trim_silence(audio.mean(axis=1), sr), sr))
        summarise(name, clips)
    for student in students:
        voice = load_voice(student)
        clips = [(trim_silence(voice.synthesize(normalize_for_speech(s)), voice.sample_rate), voice.sample_rate)
                 for s in sentences]
        summarise(student.split("amy_distill_")[-1].removesuffix("-medium"), clips)
    print("\nreference (healthy natural voice, MDVP): jitter < ~1.0 %, shimmer < ~3.8 %, HNR > ~20 dB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
