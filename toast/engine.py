"""ToastTTS inference: the one entry point for speaking your own text.

    from toast.engine import ToastEngine

    tts = ToastEngine()                        # amy at 1.2x, loaded once
    tts.say("Hi! Give me a second, let me check.")   # plays it, returns timing
    audio = tts.synthesize("Any text.")        # float32 array at tts.sample_rate
    tts.save("Any text.", "hello.wav")

    # Text arriving bit by bit (an LLM reply, a socket, a pipe):
    tts.say(token for token in llm_stream)

Everything the research settled is the default here: the "smart" splitter
(fast first piece, pauses at asides and list items, "running low" rule),
punctuation-based pauses with fades and room tone, and phrase-final slowing.
See research/README.md for why.
"""

import time
from dataclasses import dataclass

import numpy as np

from toast.pacing import stream_from_llm
from toast.stream_chunker import StreamChunker
from toast.voices import load_voice

DEFAULT_VOICE = "amy"   # research/decisions.md D16
DEFAULT_SPEED = 1.2


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
    def __init__(self, voice=DEFAULT_VOICE, speed=DEFAULT_SPEED, seed=None, open_audio=False,
                 wake_s=None, idle_db=None, attach_openers=False):
        """voice: any name load_voice() accepts ("amy", "lessac", "piper:en_US-ryan-medium",
        "kitten", ...). seed: fix it for repeatable pause lengths; None varies them.
        open_audio: open the sound device now rather than on the first say(), so
        the first utterance doesn't pay for it (~10-60 ms) and the device is
        already awake when the first word arrives. wake_s / idle_db: see
        LivePlayer (for speakers that swallow the first word). attach_openers:
        keep "Well," / "So," with the words after it (more natural, slower start;
        research/notebook/12)."""
        self.voice = load_voice(voice)
        self.speed = speed
        self.seed = seed
        self.sample_rate = self.voice.sample_rate
        self.attach_openers = attach_openers
        self._player = None
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

    def stream(self, text, pieces=None):
        """Yield audio chunks (speech, then its pause) as soon as each is ready.

        text: a string (all at once) or any iterable of strings arriving over
        time. pieces: optional list that receives each text piece as it's cut.
        """
        tokens = [text] if isinstance(text, str) else text
        chunker = (_RecordingChunker(pieces, attach_openers=self.attach_openers) if pieces is not None
                   else StreamChunker(attach_openers=self.attach_openers))
        return stream_from_llm(self.voice, tokens, speed=self.speed, seed=self.seed, chunker=chunker)

    def synthesize(self, text):
        return self.run(text, play=False).audio

    def save(self, text, path):
        import soundfile as sf

        result = self.run(text, play=False)
        sf.write(path, result.audio, self.sample_rate)
        return result

    def say(self, text):
        """Speak out loud while generating; returns a SpeechResult with timings."""
        return self.run(text, play=True)

    def run(self, text, play=True):
        """Make speech for text (string or stream of strings), optionally playing
        it live; returns a SpeechResult with the audio and timings."""
        pieces = []
        chunks = self.stream(text, pieces)
        if play:
            from toast.player import play_while_making

            chunks = play_while_making(chunks, self.sample_rate, player=self._open_player())
        start = time.perf_counter()
        audio, ttfa, made_in = [], None, 0.0
        for chunk in chunks:
            if ttfa is None:
                ttfa = (time.perf_counter() - start) * 1000
            audio.append(chunk)
            made_in = time.perf_counter() - start  # generation time, not playback time
        audio = np.concatenate(audio) if audio else np.zeros(0, dtype=np.float32)
        return SpeechResult(audio, self.sample_rate, ttfa or 0.0, made_in, pieces)


class _RecordingChunker(StreamChunker):
    def __init__(self, record, **options):
        super().__init__(**options)
        self._record = record

    def _take(self, cut):
        piece = super()._take(cut)
        self._record.append(piece[0])
        return piece
