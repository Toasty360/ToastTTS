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

import numpy as np

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


def _clean(w):
    return re.sub(r"^[^a-z0-9']+|[^a-z0-9']+$", "", w.lower())


def _feats(raw_word):
    w = raw_word.strip()
    comma = 1.0 if w.endswith(",") else 0.0
    period = 1.0 if re.search(r"[.!?]$", w) else 0.0
    other = 1.0 if (re.search(r"[;:]$", w) and not comma and not period) else 0.0
    return [min(len(_clean(w)), 20) / 20.0, comma, period, other]


def _punct_kind(raw_word):
    w = raw_word.strip()
    if w.endswith(","):
        return "comma"
    if re.search(r"[.!?]$", w):
        return "stop"
    return "other"


class OnnxPausePredictor:
    """Torch-free version of the e16 PausePredictor, running on ONNX Runtime.

    Same predict_pauses(text) contract, so it drops straight into
    ToastEngine(pause_model=...). Needs only onnxruntime + numpy — no torch
    (~210 MB import). Loads experiments/e16_pause_model/data/model/
    pause_model.onnx plus pause_meta.json (vocab, duration lookups).

    Usage:
        pred = OnnxPausePredictor("experiments/e16_pause_model/data/model")
        engine = ToastEngine(pause_model=pred)
    """

    def __init__(self, model_dir):
        import json
        from pathlib import Path

        import onnxruntime as ort

        d = Path(model_dir)
        meta = json.loads((d / "pause_meta.json").read_text())
        self.vocab = {w: i for i, w in enumerate(meta["vocab"])}
        self.punct_med_ms = meta.get("punct_med_ms")
        self.bucket_med_ms = meta.get("bucket_med_ms")
        self.sess = ort.InferenceSession(
            str(d / "pause_model.onnx"), providers=["CPUExecutionProvider"])

    def predict_pauses(self, text):
        toks = [t for t in _WORD.findall(text) if t]
        if not toks:
            return []
        ids = np.array(
            [[self.vocab.get(_clean(t), 1) for t in toks]], dtype=np.int64)
        ft = np.array([[_feats(t) for t in toks]], dtype=np.float32)
        lp, lb = self.sess.run(None, {"ids": ids, "ft": ft})
        # The ONNX graph returns full-sequence logits; the model predicts the
        # boundary AFTER word i, so drop the last timestep (no following
        # boundary), mirroring the torch model's h[:, :-1, :].
        probs = 1.0 / (1.0 + np.exp(-lp[0][:-1]))
        buckets = lb[0][:-1].argmax(-1)
        out = []
        for i, t in enumerate(toks):
            if i < len(probs):
                prob = float(probs[i])
                if prob >= 0.5:
                    if self.punct_med_ms:
                        ms = int(self.punct_med_ms[_punct_kind(t)])
                    else:
                        ms = int(self.bucket_med_ms[int(buckets[i])])
                else:
                    ms = 0
            else:  # last word: no following boundary
                prob, ms = 0.0, 0
            out.append({"word": t, "pause_prob": round(prob, 3), "pause_ms": ms})
        return out
