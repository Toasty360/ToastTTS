"""Measure and reshape the pitch melody of speech (via Praat / parselmouth).

"Robotic" speech usually means a flat, even melody: the pitch moves too
little and too slowly. These tools measure how much a voice's pitch moves
and can scale amy's pitch movement up toward a more expressive voice.
Only the melody changes; the voice itself (timbre) and timing stay the same.

Pitch is measured in semitones (st), the musical scale, because that's how
ears hear pitch differences: +12 st is an octave.
"""

import numpy as np
import parselmouth
from parselmouth.praat import call

PITCH_FLOOR, PITCH_CEILING = 75, 500  # Hz, covers typical male and female voices


def _semitones(hz, ref=100.0):
    return 12 * np.log2(hz / ref)


def pitch_profile(audio, sample_rate):
    """Summary of how the pitch moves:
      median_hz      typical pitch
      spread_st      standard deviation of pitch: how widely it varies
      range_st       5th-95th percentile span: the usual range used
      movement_st_s  average speed of pitch change while voiced (st per second)
    """
    sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sample_rate)
    pitch = sound.to_pitch_ac(time_step=0.01, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    hz = pitch.selected_array["frequency"]
    voiced = hz > 0
    st = _semitones(hz[voiced])
    # Movement: only between consecutive voiced frames (not across silences).
    both = voiced[1:] & voiced[:-1]
    steps = np.abs(np.diff(_semitones(np.where(voiced, hz, 1.0))))[both]
    return {
        "median_hz": float(np.median(hz[voiced])),
        "spread_st": float(np.std(st)),
        "range_st": float(np.percentile(st, 95) - np.percentile(st, 5)),
        "movement_st_s": float(np.mean(steps) / 0.01),
    }


def expand_pitch(audio, sample_rate, factor):
    """Scale pitch movement around the speaker's own median by `factor`
    (1.0 = unchanged, 1.5 = 50% more melody), resynthesized with Praat's
    overlap-add (PSOLA), which keeps the voice's character."""
    if factor == 1.0:
        return audio
    median = pitch_profile(audio, sample_rate)["median_hz"]
    sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sample_rate)
    manipulation = call(sound, "To Manipulation", 0.01, PITCH_FLOOR, PITCH_CEILING)
    tier = call(manipulation, "Extract pitch tier")
    call(tier, "Formula", f"{median} * exp({factor} * ln(self / {median}))")
    call([tier, manipulation], "Replace pitch tier")
    result = call(manipulation, "Get resynthesis (overlap-add)")
    return result.values[0].astype(np.float32)
