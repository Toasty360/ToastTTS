from pathlib import Path

import numpy as np
import pytest

MODEL = Path(__file__).resolve().parents[1] / "models" / "en_US-amy-medium.onnx"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="amy voice not downloaded")


@pytest.fixture(scope="module")
def tts():
    from toast.engine import ToastEngine

    return ToastEngine(seed=0)


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
