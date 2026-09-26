"""Pauses from the learned e16 pause model instead of the PAUSES_MS table.

The model predicts, for each word, how much silence follows it. We predict
once for the whole text, then hand speak_pieces a callable that looks up the
prediction for the word each piece ends on. The piece cuts still come from
the normal StreamChunker — only the pause durations change — so an A/B
against the table isolates exactly one variable.

Only for complete text: the live-LLM path doesn't know the full text yet, so
it keeps the table. ToastEngine applies it automatically when constructed
with pause_model=<predictor> and given a complete string.
"""

import re

_WORD = re.compile(r"\S+")


def learned_pauses(predictor, text):
    """Build the pause_ms callable for speak_pieces from a pause predictor.

    `predictor` is experiments/e16_pause_model's PausePredictor (any object
    with predict_pauses(text) -> [{"word", "pause_prob", "pause_ms"}] works).
    `text` must be the same text that plan_pieces will split: pieces are
    matched back to the prediction word by word, and any piece that doesn't
    line up falls back to the PAUSES_MS table.
    """
    pred = predictor.predict_pauses(text)
    toks = _WORD.findall(text)
    if len(pred) != len(toks):
        raise ValueError(f"predictor returned {len(pred)} predictions for {len(toks)} words")
    pos = 0

    def pause_ms(piece, ends_sentence):
        nonlocal pos
        words = _WORD.findall(piece)
        if toks[pos:pos + len(words)] != words:
            return None  # not the expected words: fall back to the table
        pos += len(words)
        return pred[pos - 1]["pause_ms"]

    return pause_ms
