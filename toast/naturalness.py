"""Predict how natural speech sounds, on a 1-5 scale, without a human.

UTMOS is a model trained on thousands of human ratings of speech ("how
natural does this sound, 1 to 5?"). It's not a perfect judge, but it's
consistent, so it's good for comparing voices and versions. For reference,
on our paragraph: Kitten 4.35, lessac-high 4.44, lessac-medium 3.82.
"""

import numpy as np

_model = None


def _utmos():
    global _model
    if _model is None:
        import torch

        _model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        _model.eval()
    return _model


def naturalness(audio, sample_rate):
    """Predicted rating for one clip (best for a sentence or two)."""
    import torch

    # UTMOS expects 16 kHz audio.
    times = np.linspace(0, len(audio) / sample_rate, int(len(audio) * 16000 / sample_rate), endpoint=False)
    audio16 = np.interp(times, np.arange(len(audio)) / sample_rate, audio).astype(np.float32)
    with torch.no_grad():
        return float(_utmos()(torch.from_numpy(audio16).unsqueeze(0), 16000))
