#!/usr/bin/env python3
"""Pause prediction for ToastTTS.

API for ToastEngine (import this module):
    from predict import PausePredictor
    p = PausePredictor("pause_model.pt")
    out = p.predict_pauses("Hello world, how are you today")
    # -> [{"word": "Hello", "pause_prob": 0.02, "pause_ms": 0},
    #     {"word": "world,", "pause_prob": 0.91, "pause_ms": 320}, ...]
    # pause_ms is the predicted silence AFTER the word (0 = no pause).

CLI:
    python predict.py --checkpoint pause_model.pt --text "Hello world, how are you"
"""
import argparse, json, re, sys, os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import PauseModel, clean, feats, punct_kind  # noqa: E402

def tokenize(text):
    return [t for t in re.findall(r"\S+", text) if t]

class PausePredictor:
    def __init__(self, checkpoint, device="cpu"):
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        self.vocab = ck["vocab"]
        self.bucket_med_ms = ck["bucket_med_ms"]
        # empirical pause ms per punctuation kind (falls back to bucket medians
        # for checkpoints trained before this lookup existed)
        self.punct_med_ms = ck.get("punct_med_ms")
        self.pause_ms_thresh = ck.get("pause_ms", 150.0)
        self.model = PauseModel(len(self.vocab))
        self.model.load_state_dict(ck["state"])
        self.model.eval()

    def predict_pauses(self, text):
        toks = tokenize(text)
        if not toks:
            return []
        ids = torch.tensor([[self.vocab.get(clean(t), 1) for t in toks]])
        ft = torch.tensor([[feats(t) for t in toks]], dtype=torch.float32)
        with torch.no_grad():
            lp, lb = self.model(ids, ft)
            probs = torch.sigmoid(lp)[0].tolist()
            buckets = lb.argmax(-1)[0].tolist()
        out = []
        for i, t in enumerate(toks):
            if i < len(probs):
                prob = probs[i]
                if prob >= 0.5:
                    if self.punct_med_ms:
                        ms = int(self.punct_med_ms[punct_kind(t)])
                    else:
                        ms = int(self.bucket_med_ms[buckets[i]])
                else:
                    ms = 0
            else:  # last word: no following boundary
                prob, ms = 0.0, 0
            out.append({"word": t, "pause_prob": round(prob, 3), "pause_ms": ms})
        return out

    def render(self, text):
        """Human-readable: text with [pause: NNNms] markers."""
        parts = []
        for d in self.predict_pauses(text):
            parts.append(d["word"])
            if d["pause_ms"]:
                parts.append(f"[pause: {d['pause_ms']}ms]")
        return " ".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--text", required=True)
    args = ap.parse_args()
    p = PausePredictor(args.checkpoint)
    print(p.render(args.text))
    print(json.dumps(p.predict_pauses(args.text), indent=2))

if __name__ == "__main__":
    main()
