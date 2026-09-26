"""One simple interface for every TTS model we try.

Every voice takes text and returns audio as a float32 numpy array
(values between -1 and 1), and tells you its sample rate.
Swapping models means changing one name, nothing else.
"""

import contextlib
import io
from pathlib import Path

import numpy as np

from toast.pronounce import apply_respellings

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class PiperEngine:
    def __init__(self, voice_name="en_US-lessac-medium", speaker=None,
                 noise_scale=None, noise_w_scale=None):
        from piper import PiperVoice

        self._voice = PiperVoice.load(MODELS_DIR / f"{voice_name}.onnx")
        # Two randomness knobs (None = whatever the voice file says; libritts_r
        # says 0.333 for both, which sounds flat and bored):
        #   noise_scale    randomness of the sound itself: more expressive, but
        #                  too high (0.667) brings out gasps and breathy artifacts
        #   noise_w_scale  randomness of timing: a less even, less robotic rhythm
        self._noise_scale = noise_scale
        self._noise_w_scale = noise_w_scale
        self.sample_rate = self._voice.config.sample_rate
        self.name = f"piper:{voice_name}" + (f":{speaker}" if speaker else "")
        if noise_scale is not None or noise_w_scale is not None:
            self.name += f"@{noise_scale}/{noise_w_scale}"

        # Multi-speaker models (like libritts_r) name speakers such as "3922";
        # the model itself wants their number (3922 -> 0).
        self._speaker_id = None
        if speaker is not None:
            speaker_map = self._voice.config.speaker_id_map or {}
            if speaker not in speaker_map:
                raise ValueError(f"{voice_name} has no speaker '{speaker}'")
            self._speaker_id = speaker_map[speaker]

    def synthesize(self, text, speed=1.0):
        from piper import SynthesisConfig

        # Piper's knob is "length_scale": bigger = slower, so it's 1/speed.
        config = SynthesisConfig(
            speaker_id=self._speaker_id,
            length_scale=1.0 / speed,
            noise_scale=self._noise_scale,
            noise_w_scale=self._noise_w_scale,
        )
        text = apply_respellings(text)
        pieces = [chunk.audio_float_array for chunk in self._voice.synthesize(text, config)]
        if not pieces:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(pieces).astype(np.float32)


class KittenEngine:
    sample_rate = 24000

    def __init__(self, model="KittenML/kitten-tts-mini-0.8", voice="Jasper"):
        from kittentts import KittenTTS

        with contextlib.redirect_stdout(io.StringIO()):
            self._model = KittenTTS(model)
        self._voice = voice
        self.name = f"kitten:{model.split('/')[-1]}:{voice}"

    def synthesize(self, text, speed=1.0):
        # KittenTTS prints a line on every call; keep the console clean.
        with contextlib.redirect_stdout(io.StringIO()):
            audio = self._model.generate(text, voice=self._voice, speed=speed, clean_text=True)
        return np.asarray(audio, dtype=np.float32).reshape(-1)


class KokoroEngine:
    """Kokoro-82M (StyleTTS2-based, Apache-2.0) via kokoro-onnx.

    More expressive than Piper, but a much bigger model (82M vs ~15M params).
    Kokoro adds its own pauses between sentences and clauses; they're turned
    off because ToastTTS inserts its own (toast/pacing.py).
    """

    sample_rate = 24000

    def __init__(self, voice="af_heart", precision="fp32"):
        from kokoro_onnx import Kokoro

        model = "kokoro-v1.0.onnx" if precision == "fp32" else f"kokoro-v1.0.{precision}.onnx"
        self._kokoro = Kokoro(str(MODELS_DIR / "kokoro" / model), str(MODELS_DIR / "kokoro" / "voices-v1.0.bin"))
        self._voice = voice
        self._lang = "en-gb" if voice.startswith("b") else "en-us"
        self.name = f"kokoro:{voice}" + ("" if precision == "fp32" else f":{precision}")

    def synthesize(self, text, speed=1.0):
        text = apply_respellings(text)
        audio, _ = self._kokoro.create(text, voice=self._voice, speed=speed, lang=self._lang,
                                       sentence_pause=0.0, clause_pause=0.0)
        return np.asarray(audio, dtype=np.float32).reshape(-1)


def load_voice(name):
    """name is "piper", "kitten" or "kokoro", optionally with details:
    "piper:en_US-libritts_r-medium:3922", "kitten:micro", "kitten:mini:Luna",
    "kokoro:af_heart", "kokoro:af_heart:int8".
    "amy" is the default voice (en_US-amy-medium, best at speed 1.2): natural,
    0 wrong words in our benchmark, fast. "lessac" (en_US-lessac-medium) and
    "3922" (libritts_r) are shortcuts too.
    Piper randomness can be added after "@": "3922@0.333/0.8" means
    noise_scale 0.333, noise_w_scale 0.8. "+lively" means "@0.667/0.8".
    """
    noise = (None, None)
    if name.endswith("+lively"):
        name, noise = name.removesuffix("+lively"), (0.667, 0.8)
    if "@" in name:
        name, settings = name.split("@")
        noise = tuple(float(value) for value in settings.split("/"))
    if name == "3922":
        name = "piper:en_US-libritts_r-medium:3922"
    if name == "lessac":
        name = "piper:en_US-lessac-medium"
    if name == "amy":
        name = "piper:en_US-amy-medium"
    kind, *options = name.split(":")
    if kind == "piper":
        return PiperEngine(*options, noise_scale=noise[0], noise_w_scale=noise[1])
    if kind == "kitten":
        model = f"KittenML/kitten-tts-{options[0]}-0.8" if options else "KittenML/kitten-tts-mini-0.8"
        voice = options[1] if len(options) > 1 else "Jasper"
        return KittenEngine(model, voice)
    if kind == "kokoro":
        return KokoroEngine(*options)  # "kokoro", "kokoro:af_bella", "kokoro:af_heart:int8"
    raise ValueError(f"Unknown voice '{name}'. Use 'piper', 'kitten' or 'kokoro'.")
