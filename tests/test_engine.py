import threading
import time
from pathlib import Path

import numpy as np
import pytest

MODEL = Path(__file__).resolve().parents[1] / "models" / "en_US-amy-medium.onnx"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="amy voice not downloaded")


@pytest.fixture(scope="module")
def tts():
    from toast.engine import ToastEngine

    return ToastEngine("amy", seed=0)


def test_speaks_a_string(tts):
    result = tts.run("Well, hello there. How are you?", play=False)
    assert result.audio.dtype == np.float32 and result.seconds > 1.0
    assert result.pieces == ["Well,", "hello there.", "How are you?"]
    assert result.ttfa_ms > 0


def test_streamed_text_gives_the_same_pieces(tts):
    tokens = ["We", "ll,", " hel", "lo", " there.", " How", " are", " you?"]
    assert tts.run(iter(tokens), play=False).pieces == ["Well,", "hello there.", "How are you?"]


def test_empty_text_gives_no_audio(tts):
    result = tts.run("", play=False)
    assert len(result.audio) == 0 and result.pieces == []


class FakePlayer:
    """Stands in for the sound card: tests decide how much has been played."""

    def __init__(self):
        self.played = self.queued = 0

    def add(self, chunk):
        start, self.queued = self.queued, self.queued + len(chunk)
        return start

    def clear(self, fade_ms=20):
        self.queued = self.played

    def wait(self):
        self.played = self.queued


def test_stop_cuts_a_reply_off_and_reports_what_was_heard(tts):
    tts._player = player = FakePlayer()
    go_on = threading.Event()

    def llm():
        yield "Well, hello there, "
        go_on.wait(5)
        yield "how are you today?"

    try:
        speech = tts.speak(llm())
        while not speech._spans:  # "Well," is made and queued
            time.sleep(0.01)
        player.played = player.queued
        speech.stop()
        go_on.set()
        speech.wait()
        assert speech.heard() == "Well,"
        assert speech.done and "how are you today?" not in speech.pieces
    finally:
        tts._player = None
