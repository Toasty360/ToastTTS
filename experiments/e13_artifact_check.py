"""E13 follow-up: are kNN-VC's extra pitch range and 16 kHz output problems?

1. Pitch glitches: share of voiced frames where pitch jumps more than
   6 semitones in 10 ms (a real voice glides; octave jumps are artifacts).
   Also the pitch range with those frames' outliers ignored (10th-90th pct).
2. Bandwidth: share of energy above 7 kHz (kNN-VC can't have any above
   8 kHz; amy's model trains on audio up to 11 kHz).

  .venv\\Scripts\\python experiments\\e13_artifact_check.py 20260925-2151
"""

import sys
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def analyse(clips):
    jumps, voiced_frames, semitones, high, total = 0, 0, [], 0.0, 0.0
    for audio, sr in clips:
        pitch = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr).to_pitch_ac(
            time_step=0.01, pitch_floor=75, pitch_ceiling=500)
        hz = pitch.selected_array["frequency"]
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
    semitones = np.array(semitones)
    return (100 * jumps / max(voiced_frames, 1), np.percentile(semitones, 90) - np.percentile(semitones, 10),
            100 * high / total)


def main(run_id):
    run = ROOT / "out" / "e13" / run_id
    print(f"{'condition':<15}{'pitch jumps %':>14}{'range 10-90 st':>16}{'energy >7kHz %':>16}")
    for name in ("kokoro_source", "amy_direct", "knnvc", "openvoice"):
        clips = []
        for path in sorted((run / name).glob("clip_*.wav")):
            audio, sr = sf.read(path, dtype="float32", always_2d=True)
            clips.append((audio.mean(axis=1), sr))
        jumps, core_range, high = analyse(clips)
        print(f"{name:<15}{jumps:>14.2f}{core_range:>16.2f}{high:>16.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
