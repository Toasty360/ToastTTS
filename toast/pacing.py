"""Turn text into speech with human-like pauses, piece by piece.

The model speaks one piece at a time. Between pieces we insert our own
pauses, sized by the punctuation, instead of trusting the model (Piper
barely pauses at commas; Kitten pads everything with silence).

For every piece:
  1. trim the model's own leading/trailing silence
  2. fade the edges so nothing clicks
  3. add a pause whose length depends on the punctuation, with a little
     randomness so it never sounds like a metronome
Quiet "room tone" runs under everything, so pauses sound like a person in
a room, not a recording that switched off.

Audio comes out as a stream (a generator), so the first piece can start
playing while the rest is still being made.
"""

import time

import numpy as np

from toast.splitter import ending_mark, split_into_pieces, split_into_sentences
from toast.stream_chunker import StreamChunker, is_opener
from toast.text_normalize import normalize_for_speech

# Pause after each kind of punctuation, in milliseconds (shortest, longest).
PAUSES_MS = {
    ",": (180, 260),
    ";": (300, 380),
    ":": (280, 360),
    "—": (260, 360),
    "...": (500, 700),
    ".": (480, 620),
    "!": (450, 580),
    "?": (520, 680),
    "": (60, 120),  # a cut with no punctuation (fast first piece, run-ons)
}

# One-word openers ("Well,", "So,", "Okay,") get a shorter pause than other
# commas: natural voices pause 80-320 ms after "Well," (measured with Whisper
# gaps), amy measured 420 ms with the normal comma pause (research/notebook/12).
OPENER_PAUSE_MS = (60, 140)

# People slow down at the end of a sentence ("phrase-final lengthening").
FINAL_PHRASE_SPEED = 0.94


# When less audio than this is waiting to be played, the live splitter stops
# waiting for the ideal pause point (see StreamChunker.running_low).
LOW_QUEUE_SECONDS = 0.4


# Phrase-final softening: natural voices get quieter on the last word before
# a pause instead of stopping at full volume (research/notebook/11). Over the
# last SOFT_TAIL_MS of every piece the volume eases down by SOFT_TAIL_DB.
SOFT_TAIL_MS = 500   # set with experiments/e10 to match Kokoro / Soniox (E09)
SOFT_TAIL_DB = -6.0

FADE_MS = 12
ROOM_TONE_DB = -60  # very quiet: felt more than heard


def trim_silence(audio, sample_rate, threshold_db=-45, keep_start_ms=10, keep_end_ms=25):
    """Cut quiet audio off both ends, keeping a short margin so soft sounds
    like a final "s" or "t" are not chopped. The start margin is smaller:
    silence there is heard as delay before speech begins."""
    if len(audio) == 0:
        return audio
    frame = int(sample_rate * 0.01)
    usable = len(audio) // frame * frame
    rms = np.sqrt(np.mean(audio[:usable].reshape(-1, frame) ** 2, axis=1) + 1e-12)
    loud = np.nonzero(20 * np.log10(rms / (np.abs(audio).max() + 1e-12)) > threshold_db)[0]
    if len(loud) == 0:
        return audio[:0]
    start = max(0, loud[0] * frame - int(sample_rate * keep_start_ms / 1000))
    end = min(len(audio), (loud[-1] + 1) * frame + int(sample_rate * keep_end_ms / 1000))
    return audio[start:end]


def fade_edges(audio, sample_rate, fade_ms=FADE_MS):
    """Smoothly ramp the volume up at the start and down at the end."""
    n = min(int(sample_rate * fade_ms / 1000), len(audio) // 2)
    if n == 0:
        return audio
    audio = audio.copy()
    ramp = np.sin(np.linspace(0, np.pi / 2, n)) ** 2  # gentle S-shaped curve
    audio[:n] *= ramp
    audio[-n:] *= ramp[::-1]
    return audio


def soften_ending(audio, sample_rate, tail_ms=None, depth_db=None):
    """Ease the volume down over the end of a piece (roughly its last word),
    so speech settles into the pause like a person's does."""
    tail_ms = SOFT_TAIL_MS if tail_ms is None else tail_ms
    depth_db = SOFT_TAIL_DB if depth_db is None else depth_db
    # At most the last 30% of the piece, so a one-word piece ("Well,") isn't
    # faded through its middle.
    n = min(int(sample_rate * tail_ms / 1000), int(len(audio) * 0.3))
    if n == 0 or depth_db == 0:
        return audio
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))  # smooth 0 -> 1
    audio = audio.copy()
    audio[-n:] *= 10 ** (depth_db * ramp / 20)
    return audio


class RoomTone:
    """Soft, low hiss like a quiet room: random noise with the harsh highs
    smoothed away. Made once, then looped, so it costs nothing per piece."""

    def __init__(self, sample_rate, level_db=None, seconds=2.0, seed=0):
        level_db = ROOM_TONE_DB if level_db is None else level_db
        rng = np.random.default_rng(seed)
        n = int(sample_rate * seconds)
        smooth = max(1, int(sample_rate / 2000))  # averaging over ~0.5 ms keeps the low rumble
        noise = np.convolve(rng.standard_normal(n + 2 * smooth), np.ones(smooth) / smooth, mode="same")
        noise = noise[smooth:smooth + n]
        noise *= 10 ** (level_db / 20) / (np.sqrt(np.mean(noise ** 2)) + 1e-12)
        self._loop = noise.astype(np.float32)
        self._position = 0

    def next(self, num_samples):
        index = (self._position + np.arange(num_samples)) % len(self._loop)
        self._position = (self._position + num_samples) % len(self._loop)
        return self._loop[index]


def _text_for_model(piece):
    # A piece ending in ; : or a dash is mid-sentence: the thought continues.
    # Piper rushes the last word before ";" ("semicolon;" loses its "-lon"),
    # but finishes it fully before ",". We add the real pause ourselves, so
    # the model only needs to hear "more is coming".
    piece = normalize_for_speech(piece)
    if piece.endswith((";", ":", "—", "–")):
        return piece.rstrip(";:—–").rstrip() + ","
    return piece


def plan_pieces(text, split="smart"):
    """Return (piece, ends_sentence) pairs for text that is already complete.

    split="clause"   cut at every comma/semicolon/dash (most pauses)
    split="sentence" cut only between sentences (smoothest intonation, slow start)
    split="smart"    the live splitter's rules: cut at sentence ends, ; : — ...,
                     and at commas after short pieces like "Well," or "12,"
    """
    if split == "smart":
        chunker = StreamChunker()
        return chunker.feed(text) + chunker.finish()
    plan = []
    for sentence in split_into_sentences(text):
        pieces = split_into_pieces(sentence) if split == "clause" else [sentence]
        plan += [(piece, i == len(pieces) - 1) for i, piece in enumerate(pieces)]
    return plan


def speak_pieces(voice, pieces, speed=1.0, seed=0, final_slowdown=True):
    """Yield audio chunks: a spoken piece, then its pause, for each piece.

    `pieces` can arrive over time (from a live LLM), which is why every piece
    is followed by its pause right away: we can't know yet if it's the last.
    """
    rng = np.random.default_rng(seed)
    sr = voice.sample_rate
    room = RoomTone(sr, seed=seed)
    starts_sentence = True

    for piece, ends_sentence in pieces:
        # Slow the last piece of a sentence that was split into several pieces.
        slow_down = final_slowdown and ends_sentence and not starts_sentence
        audio = voice.synthesize(_text_for_model(piece), speed=speed * (FINAL_PHRASE_SPEED if slow_down else 1.0))
        speech = fade_edges(soften_ending(trim_silence(audio, sr), sr), sr)
        yield np.clip(speech + room.next(len(speech)), -1.0, 1.0)

        low, high = OPENER_PAUSE_MS if is_opener(piece) else PAUSES_MS.get(ending_mark(piece), PAUSES_MS[""])
        yield room.next(int(sr * rng.uniform(low, high) / 1000))
        starts_sentence = ends_sentence


def stream_speech(voice, text, split="smart", speed=1.0, seed=0):
    """Speech for text that is already complete."""
    return speak_pieces(voice, plan_pieces(text, split), speed, seed, final_slowdown=split != "sentence")


def stream_from_llm(voice, tokens, speed=1.0, seed=0, chunker=None):
    """Speech for text that arrives bit by bit, e.g. an LLM reply.

    Playback starts with the first chunk, so at any moment we know how much
    audio is still queued ahead of the listener. When that gets low, the
    chunker is told to hurry.
    """
    chunker = chunker or StreamChunker()
    first_audio_at = None
    audio_made = 0.0

    def running_low():
        if first_audio_at is None:
            return False
        queued = audio_made - (time.perf_counter() - first_audio_at)
        return queued < LOW_QUEUE_SECONDS

    chunker.running_low = running_low

    def pieces():
        for token in tokens:
            yield from chunker.feed(token)
        yield from chunker.finish()

    for chunk in speak_pieces(voice, pieces(), speed, seed):
        if first_audio_at is None:
            first_audio_at = time.perf_counter()
        audio_made += len(chunk) / voice.sample_rate
        yield chunk


def speak_naturally(voice, text, split="smart", speed=1.0, seed=0):
    """The whole text as one audio array."""
    return np.concatenate(list(stream_speech(voice, text, split, speed, seed)))
