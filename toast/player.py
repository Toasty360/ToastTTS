"""Play audio while it is still being made.

The sound card asks for small blocks of audio every few milliseconds, on
its own thread. We keep a queue of chunks; the sound card takes from the
front while new chunks are added at the back. If the queue runs dry,
it plays silence until more arrives (that is a "stall").

The output device is opened once and kept running: opening it takes
~10-60 ms, which would otherwise be added to every utterance's time to
first audio. "low" latency keeps the sound card's own buffer small, so
audio is heard sooner after it's queued.

When idle it plays very quiet room tone (a -60 dB hiss; speech itself has
none since D44) rather than exact digital silence. Some outputs,
especially monitor/TV speakers over HDMI or DisplayPort and some Bluetooth
devices, mute on digital silence and take hundreds of ms to wake up, which
swallows the first word (research/notebook/09).
"""

import collections
import threading
import time

import numpy as np

DEVICE_WAKE_S = 0.6  # how long a freshly opened device runs before the first word


class LivePlayer:
    def __init__(self, sample_rate, latency="low", keep_awake=True, idle_db=None, wake_s=DEVICE_WAKE_S):
        """idle_db: loudness of the keep-awake room tone (default -60 dB; was the level
        under speech). wake_s: how long a freshly opened device runs before the
        first word; some monitors/TVs need more than the default."""
        import sounddevice as sd

        from toast.pacing import RoomTone

        self.sample_rate = sample_rate
        self._chunks = collections.deque()
        self._current = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        # Sample counts of queued audio (idle tone not included): what the
        # sound card has taken so far, and where the next added chunk starts.
        self.played = 0
        self.queued = 0
        self._idle = RoomTone(sample_rate, level_db=idle_db, seed=1) if keep_awake else None
        self.wake_s = wake_s
        self._stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32",
                                       latency=latency, callback=self._fill)
        self._stream.start()
        self.started_at = time.perf_counter()

    def add(self, chunk):
        """Queue a chunk; returns the sample position where it will start."""
        with self._lock:
            self._chunks.append(np.asarray(chunk, dtype=np.float32))
            start = self.queued
            self.queued += len(chunk)
            return start

    def clear(self, fade_ms=20):
        """Barge-in: fade out what's playing within fade_ms and drop the rest."""
        with self._lock:
            n = min(len(self._current), int(self.sample_rate * fade_ms / 1000))
            self._current = self._current[:n] * np.linspace(1, 0, n, dtype=np.float32)
            self._chunks.clear()
            self.queued = self.played + n

    def _fill(self, out, frames, time_info, status):
        written = 0
        with self._lock:
            while written < frames:
                if len(self._current) == 0:
                    if not self._chunks:
                        break
                    self._current = self._chunks.popleft()
                take = min(frames - written, len(self._current))
                out[written:written + take, 0] = self._current[:take]
                self._current = self._current[take:]
                written += take
                self.played += take
        out[written:, 0] = self._idle.next(frames - written) if self._idle else 0.0

    def wait_until_awake(self):
        """Only needed right after opening: let a sleepy output wake up."""
        remaining = self.wake_s - (time.perf_counter() - self.started_at)
        if remaining > 0:
            time.sleep(remaining)

    def wait(self):
        """Block until everything queued has been played."""
        while True:
            with self._lock:
                if not self._chunks and len(self._current) == 0:
                    break
            time.sleep(0.02)
        time.sleep(self._stream.latency + 0.05)  # the sound card's own buffer

    def close(self):
        self._stream.stop()
        self._stream.close()


def play(audio, sample_rate):
    """Play a finished recording, without losing its first word."""
    player = LivePlayer(sample_rate)
    try:
        player.wait_until_awake()
        player.add(audio)
        player.wait()
    finally:
        player.close()


def play_while_making(chunks, sample_rate, player=None):
    """Pass chunks through unchanged, playing each one as it arrives.

    With a `player`, it is reused and left open; otherwise one is opened for
    this call and closed at the end.
    """
    own = player is None
    player = player or LivePlayer(sample_rate)
    try:
        for chunk in chunks:
            player.add(chunk)
            yield chunk
        player.wait()
    finally:
        if own:
            player.close()
