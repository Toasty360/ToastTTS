"""ToastTTS inference: the one entry point for speaking your own text.

    from toast.engine import ToastEngine

    tts = ToastEngine()                        # Daniel (Kokoro) at 1.2x; ToastEngine("amy") is the fast one
    tts.say("Hi! Give me a second, let me check.")   # plays it, returns timing
    audio = tts.synthesize("Any text.")        # float32 array at tts.sample_rate
    tts.save("Any text.", "hello.wav")

    # Text arriving bit by bit (an LLM reply, a socket, a pipe):
    tts.say(token for token in llm_stream)

    # In a conversation: speak in the background, stop when the user talks.
    reply = tts.speak(token for token in llm_stream)
    reply.stop()                               # barge-in: silent within ~20 ms
    reply.heard()                              # the words the user actually heard

Everything the research settled is the default here: the "smart" splitter
(fast first piece, pauses at asides and list items, "running low" rule),
punctuation-based pauses with fades, and phrase-final slowing.
See research/README.md for why.
"""

import threading
import time
from dataclasses import dataclass

import numpy as np

from toast.pacing import LOW_QUEUE_SECONDS, stream_from_llm
from toast.stream_chunker import StreamChunker
from toast.voices import load_voice

# Tuned settings per voice. low_queue_s: hurry the splitter when less audio
# than this is queued; Daniel is ~9x slower to make than amy, so it needs a
# bigger margin (0.4 s gave a stall at 10 tokens/s, 0.8 s none; D46).
PROFILES = {
    "daniel": {"speed": 1.2, "low_queue_s": 0.8},   # Kokoro bm_daniel: more natural, ~200 ms start
    "amy": {"speed": 1.2, "low_queue_s": 0.4},      # Piper amy-medium: ~60x real time, ~85 ms start
}
DEFAULT_VOICE = "daniel"   # research/decisions.md D45
DEFAULT_SPEED = 1.2        # for voices without a profile


@dataclass
class SpeechResult:
    audio: np.ndarray
    sample_rate: int
    ttfa_ms: float        # from the call (or first token) to the first chunk being ready
    made_in_s: float      # total time to produce all audio
    pieces: list          # the text pieces, as the splitter cut them

    @property
    def seconds(self):
        return len(self.audio) / self.sample_rate


class ToastEngine:
    def __init__(self, voice=DEFAULT_VOICE, speed=None, seed=None, open_audio=False,
                 wake_s=None, idle_db=None, attach_openers=False, pause_model=None, threads=None,
                 player=None):
        """voice: any name load_voice() accepts ("amy", "lessac", "piper:en_US-ryan-medium",
        "kitten", ...). seed: fix it for repeatable pause lengths; None varies them.
        open_audio: open the sound device now rather than on the first say(), so
        the first utterance doesn't pay for it (~10-60 ms) and the device is
        already awake when the first word arrives. wake_s / idle_db: see
        LivePlayer (for speakers that swallow the first word). attach_openers:
        keep "Well," / "So," with the words after it (more natural, slower start;
        research/notebook/12). pause_model: None (default) uses the hand-written
        PAUSES_MS table in toast/pacing.py. Pass a pause predictor (anything with
        predict_pauses(text), e.g. experiments/e16_pause_model's PausePredictor)
        to use learned pauses instead. The model only applies when the full text
        is known up front; live token streams and any misaligned piece fall back
        to the table. speed: None uses the voice's profile (PROFILES).
        threads: cap the CPU threads speech may use, to leave room for a
        speech recognizer and an LLM on the same machine. player: play through
        something else than the sound card, e.g. an echo-cancelling audio unit
        that must see what the speakers play. It needs LivePlayer's add(chunk)
        -> start sample, clear(fade_ms), played, wait() and close()."""
        profile = PROFILES.get(voice, {})
        self.voice = load_voice(voice, threads=threads)
        self.speed = speed or profile.get("speed", DEFAULT_SPEED)
        self.low_queue_s = profile.get("low_queue_s", LOW_QUEUE_SECONDS)
        self.seed = seed
        self.sample_rate = self.voice.sample_rate
        self.attach_openers = attach_openers
        self.pause_model = pause_model
        self._player = player
        self._audio_options = {k: v for k, v in (("wake_s", wake_s), ("idle_db", idle_db)) if v is not None}
        self.voice.synthesize("Warm up.")  # the first call is always slower
        if open_audio:
            self._open_player()

    def _open_player(self):
        if self._player is None:
            from toast.player import LivePlayer

            self._player = LivePlayer(self.sample_rate, **self._audio_options)
            # Give sleepy outputs (HDMI monitors, Bluetooth) time to wake before
            # the first word. Only once; afterwards the idle room tone keeps
            # them awake.
            self._player.wait_until_awake()
        return self._player

    def close(self):
        if self._player is not None:
            self._player.close()
            self._player = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def stream(self, text, pieces=None, on_piece=None):
        """Yield audio chunks (speech, then its pause) as soon as each is ready.

        text: a string (all at once) or any iterable of strings arriving over
        time. pieces: optional list that receives each text piece as it's cut.
        on_piece: called with a piece's text just before its speech chunk.
        """
        tokens = [text] if isinstance(text, str) else text
        chunker = (_RecordingChunker(pieces, attach_openers=self.attach_openers) if pieces is not None
                   else StreamChunker(attach_openers=self.attach_openers))
        pause_ms = None
        if isinstance(text, str) and self.pause_model is not None:
            from toast.learned_pauses import learned_pauses
            pause_ms = learned_pauses(self.pause_model, text)
        return stream_from_llm(self.voice, tokens, speed=self.speed, seed=self.seed,
                               chunker=chunker, pause_ms=pause_ms, on_piece=on_piece,
                               low_queue_s=self.low_queue_s)

    def synthesize(self, text):
        return self.run(text, play=False).audio

    def save(self, text, path):
        import soundfile as sf

        result = self.run(text, play=False)
        sf.write(path, result.audio, self.sample_rate)
        return result

    def say(self, text):
        """Speak out loud while generating; returns a SpeechResult with timings."""
        return self.speak(text).wait()

    def speak(self, text):
        """Start speaking in the background and return at once with a Speech
        handle: stop() it on barge-in, ask it what was heard()."""
        return Speech(self, text)

    def run(self, text, play=True):
        """Make speech for text (string or stream of strings), optionally playing
        it live; returns a SpeechResult with the audio and timings."""
        if play:
            return self.say(text)
        pieces = []
        chunks = self.stream(text, pieces)
        start = time.perf_counter()
        audio, ttfa, made_in = [], None, 0.0
        for chunk in chunks:
            if ttfa is None:
                ttfa = (time.perf_counter() - start) * 1000
            audio.append(chunk)
            made_in = time.perf_counter() - start  # generation time, not playback time
        audio = np.concatenate(audio) if audio else np.zeros(0, dtype=np.float32)
        return SpeechResult(audio, self.sample_rate, ttfa or 0.0, made_in, pieces)


class Speech:
    """A reply being spoken in the background (ToastEngine.speak).

    stop() cuts it off for barge-in: the voice fades out within ~20 ms and no
    more audio is made. Cancelling the LLM itself is the caller's job.
    heard() returns the words the listener actually got, for trimming the
    assistant's turn in the LLM's history.
    """

    def __init__(self, engine, text):
        self._player = engine._open_player()
        self._sample_rate = engine.sample_rate
        self._lock = threading.Lock()
        self._stopped = False
        self._spans = []    # (start sample, end sample, piece) of each spoken piece
        self._end = 0       # where this reply's audio ends in the player
        self._audio, self._error = [], None
        self.pieces, self.ttfa_ms, self.made_in_s = [], None, 0.0
        self._start = time.perf_counter()
        self._thread = threading.Thread(target=self._make, args=(engine, text), daemon=True)
        self._thread.start()

    def _make(self, engine, text):
        next_piece = []
        chunks = engine.stream(text, self.pieces, on_piece=next_piece.append)
        try:
            for chunk in chunks:
                with self._lock:
                    if self._stopped:
                        break
                    start = self._player.add(chunk)
                self._end = start + len(chunk)
                if next_piece:  # the chunk right after on_piece is that piece's speech
                    self._spans.append((start, self._end, next_piece.pop()))
                if self.ttfa_ms is None:
                    self.ttfa_ms = (time.perf_counter() - self._start) * 1000
                self._audio.append(chunk)
                self.made_in_s = time.perf_counter() - self._start
        except Exception as error:  # handed to wait(), not lost in this thread
            self._error = error
        finally:
            chunks.close()

    def stop(self):
        with self._lock:
            self._stopped = True
            self._player.clear()

    @property
    def done(self):
        """True once stopped, or once everything was made and played."""
        return self._stopped or (not self._thread.is_alive() and self._player.played >= self._end)

    def heard(self):
        played = self._player.played
        words = []
        for start, end, piece in self._spans:
            if played >= end:
                words += piece.split()
            elif played > start:
                # ponytail: assumes words are spread evenly through the piece;
                # use the model's word timings if cuts land mid-word too often.
                part = piece.split()
                words += part[:int(len(part) * (played - start) / (end - start))]
        return " ".join(words)

    def wait(self):
        """Block until made and played (or stopped); returns a SpeechResult."""
        self._thread.join()
        if self._error is not None:
            raise self._error
        self._player.wait()
        audio = np.concatenate(self._audio) if self._audio else np.zeros(0, dtype=np.float32)
        return SpeechResult(audio, self._sample_rate, self.ttfa_ms or 0.0, self.made_in_s, self.pieces)


class _RecordingChunker(StreamChunker):
    def __init__(self, record, **options):
        super().__init__(**options)
        self._record = record

    def _take(self, cut):
        piece = super()._take(cut)
        self._record.append(piece[0])
        return piece
