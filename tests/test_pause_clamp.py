"""Tests for the pause clamp (toast/pacing.py): the opt-in guardrail that caps
mid-phrase pauses without touching anything else."""

import numpy as np

from toast.pacing import clamp_pauses, find_silences


def _clip(*parts, sr=22050):
    return np.concatenate(parts).astype(np.float32), sr


def _tone(seconds, sr=22050, freq=220.0):
    n = int(sr * seconds)
    return (0.5 * np.sin(2 * np.pi * freq * np.arange(n) / sr)).astype(np.float32)


def _sil(seconds, sr=22050):
    return np.zeros(int(sr * seconds), dtype=np.float32)


def test_clamp_shrinks_only_long_internal_pauses():
    y, sr = _clip(_tone(0.5), _sil(0.5), _tone(0.5), _sil(0.1), _tone(0.5))
    out = clamp_pauses(y, sr, max_pause_ms=250)
    # 500 ms -> 250 ms (250 ms removed); the 100 ms pause is untouched.
    # Total: 0.5*3 + 0.5 + 0.1 = 2.1 s, minus 0.25 s removed = 1.85 s.
    assert abs(len(out) / sr - 1.85) < 0.02


def test_clamp_never_touches_edges():
    y, sr = _clip(_sil(0.4), _tone(0.5), _sil(0.4))
    out = clamp_pauses(y, sr, max_pause_ms=250)
    assert len(out) == len(y)  # leading/trailing silence is not clamped


def test_clamp_is_off_by_default_semantics():
    y, sr = _clip(_tone(0.5), _sil(0.5), _tone(0.5))
    assert clamp_pauses(y, sr, max_pause_ms=None) is y
    assert clamp_pauses(y, sr, max_pause_ms=0) is y


def test_clamp_no_clicks_at_join():
    y, sr = _clip(_tone(0.5), _sil(0.5), _tone(0.5))
    out = clamp_pauses(y, sr, max_pause_ms=250)
    assert np.abs(np.diff(out)).max() < 0.2  # tone amplitude is 0.5


def test_clamp_preserves_speech_bit_identical_outside_cut():
    rng = np.random.default_rng(0)
    speech = rng.standard_normal(22050).astype(np.float32) * 0.5
    y, sr = _clip(speech, _sil(0.5), speech)
    out = clamp_pauses(y, sr, max_pause_ms=250)
    # Everything before the long silence is untouched.
    assert np.array_equal(out[:22050], y[:22050])


def test_find_silences_finds_nothing_in_tone():
    y, sr = _tone(1.0), 22050
    assert find_silences(y, sr) == []
