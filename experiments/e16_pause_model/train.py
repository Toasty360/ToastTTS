#!/usr/bin/env python3
"""Train tiny pause model: word sequence -> per-boundary pause labels.

Multi-task: (a) binary pause/no-pause, (b) duration bucket classification.
Model: embedding + 2-layer BiLSTM + 2 heads. CPU, minutes.
Saves checkpoint with vocab + bucket->ms mapping for predict.py.
"""
import argparse, json, math, os, random, re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PAUSE_MS = 150.0
BUCKETS = [(0, 150), (150, 400), (400, 800), (800, float("inf"))]  # ms

def clean(w):
    return re.sub(r"^[^a-z0-9']+|[^a-z0-9']+$", "", w.lower())

def feats(raw_word):
    w = raw_word.strip()
    comma = 1.0 if w.endswith(",") else 0.0
    period = 1.0 if re.search(r"[.!?]$", w) else 0.0
    other = 1.0 if (re.search(r"[;:]$", w) and not comma and not period) else 0.0
    return [min(len(clean(w)), 20) / 20.0, comma, period, other]

def punct_kind(raw_word):
    """Coarse punctuation class of a token, for empirical pause-duration lookup."""
    w = raw_word.strip()
    if w.endswith(","):
        return "comma"
    if re.search(r"[.!?]$", w):
        return "stop"
    return "other"

class PauseDataset(Dataset):
    def __init__(self, path, vocab=None, min_count=3):
        self.exs = [json.loads(l) for l in open(path)]
        if vocab is None:
            counts = {}
            for e in self.exs:
                for w in e["words"]:
                    c = clean(w)
                    counts[c] = counts.get(c, 0) + 1
            vocab = {"<pad>": 0, "<unk>": 1}
            for w, c in sorted(counts.items(), key=lambda kv: -kv[1]):
                if c >= min_count and w:
                    vocab[w] = len(vocab)
        self.vocab = vocab
        self.items = []
        for e in self.exs:
            ids = [vocab.get(clean(w), 1) for w in e["words"]]
            ft = [feats(w) for w in e["words"]]
            gaps = e["gaps_ms"]
            pause = [1.0 if g >= PAUSE_MS else 0.0 for g in gaps]
            bucket = []
            for g in gaps:
                for bi, (lo, hi) in enumerate(BUCKETS):
                    if lo <= g < hi:
                        bucket.append(bi); break
            self.items.append((ids, ft, pause, bucket, gaps))
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate(batch):
    n = max(len(ids) for ids, _, _, _, _ in batch)
    def pad(seqs, v=0.0, d=1, length=None):
        import torch as T
        L = n if length is None else length
        t = T.full((len(seqs), L) + ((d,) if d > 1 else ()), v)
        for i, s in enumerate(seqs):
            t[i, :len(s)] = T.tensor(s, dtype=t.dtype).reshape(-1, *t.shape[2:])
        return t
    ids = pad([b[0] for b in batch])
    ft = torch.stack([torch.nn.functional.pad(
        torch.tensor(b[1]), (0, 0, 0, n - len(b[1]))) for b in batch])
    pause = pad([b[2] for b in batch], length=n - 1)
    bucket = pad([b[3] for b in batch], length=n - 1).long()
    mask = torch.zeros(len(batch), n - 1)
    for i, (ids_, _, _, _, _) in enumerate(batch):
        mask[i, :len(ids_) - 1] = 1.0
    return ids.long(), ft, pause, bucket, mask

class PauseModel(nn.Module):
    def __init__(self, vocab_size, emb=24, hid=48):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb + 4, hid, num_layers=2, bidirectional=True,
                            batch_first=True)
        self.head_pause = nn.Linear(hid * 2, 1)
        self.head_bucket = nn.Linear(hid * 2, len(BUCKETS))
    def forward(self, ids, ft):
        x = torch.cat([self.emb(ids), ft], dim=-1)
        h, _ = self.lstm(x)
        hb = h[:, :-1, :]  # boundary AFTER word i -> state at i
        return self.head_pause(hb).squeeze(-1), self.head_bucket(hb)
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

def evaluate(model, loader):
    model.eval()
    tp = fp = fn = 0
    b_ok = b_tot = 0
    ae, ae_n = 0.0, 0
    with torch.no_grad():
        for ids, ft, pause, bucket, mask in loader:
            lp, lb = model(ids, ft)
            pred = (torch.sigmoid(lp) > 0.5).float() * mask
            t = pause * mask
            tp += ((pred == 1) & (t == 1)).sum().item()
            fp += ((pred == 1) & (t == 0)).sum().item()
            fn += ((pred == 0) & (t == 1)).sum().item()
            bp = lb.argmax(-1)
            b_ok += (((bp == bucket) * mask).sum().item())
            b_tot += mask.sum().item()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"pause_p": prec, "pause_r": rec, "pause_f1": f1,
            "bucket_acc": b_ok / max(b_tot, 1), "tp": tp, "fp": fp, "fn": fn}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    tr = PauseDataset(args.train); va = PauseDataset(args.val, vocab=tr.vocab)
    tl = DataLoader(tr, batch_size=args.bs, shuffle=True, collate_fn=collate)
    vl = DataLoader(va, batch_size=args.bs, collate_fn=collate)

    # bucket -> ms mapping = median train gap within bucket
    bmed = []
    for bi, (lo, hi) in enumerate(BUCKETS):
        gs = [g for e in tr.exs for g in e["gaps_ms"] if lo <= g < hi]
        gs.sort(); bmed.append(gs[len(gs)//2] if gs else (lo + min(hi, lo+600)) / 2)
    # empirical pause duration per punctuation kind (data-driven lookup;
    # bucket distributions for comma vs period overlap too much to separate)
    pmed = {}
    for kind in ("comma", "stop", "other"):
        gs = sorted(g for e in tr.exs
                    for w, g in zip(e["words"][:-1], e["gaps_ms"])
                    if punct_kind(w) == kind and g >= PAUSE_MS)
        pmed[kind] = gs[len(gs) // 2] if gs else None
    allp = sorted(g for e in tr.exs for g in e["gaps_ms"] if g >= PAUSE_MS)
    overall = allp[len(allp) // 2] if allp else 400.0
    for k in pmed:
        if pmed[k] is None:
            pmed[k] = overall
    pos = sum(p for _, _, ps, _, _ in tr.items for p in ps)
    neg = sum(1 for _, _, ps, _, _ in tr.items for p in ps) - pos
    pos_w = neg / max(pos, 1)
    print(f"train files={len(tr)} val={len(va)} vocab={len(tr.vocab)} "
          f"pos_rate={pos/max(pos+neg,1):.3f} pos_weight={pos_w:.1f} bucket_med_ms={bmed}")
    print(f"punct_med_ms={pmed}")

    model = PauseModel(len(tr.vocab))
    print(f"params: {model.n_params()}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w), reduction="none")
    ce = nn.CrossEntropyLoss(reduction="none")

    best, best_state = -1, None
    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for ids, ft, pause, bucket, mask in tl:
            opt.zero_grad()
            lp, lb = model(ids, ft)
            loss = ((bce(lp, pause) * mask).sum() / mask.sum().clamp_min(1)
                    + (ce(lb.reshape(-1, len(BUCKETS)), bucket.reshape(-1))
                        .view_as(mask) * mask).sum() / mask.sum().clamp_min(1))
            loss.backward(); opt.step(); tot += loss.item()
        m = evaluate(model, vl)
        print(f"ep{ep+1:02d} loss={tot/len(tl):.3f} "
              f"val P={m['pause_p']:.3f} R={m['pause_r']:.3f} F1={m['pause_f1']:.3f} "
              f"bucket_acc={m['bucket_acc']:.3f}", flush=True)
        if m["pause_f1"] > best:
            best, best_state = m["pause_f1"], {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    final = evaluate(model, vl)
    ckpt = {"state": best_state, "vocab": tr.vocab, "bucket_med_ms": bmed,
            "punct_med_ms": pmed,
            "buckets": BUCKETS, "pause_ms": PAUSE_MS,
            "metrics": {"val": final, "best_f1": best}}
    torch.save(ckpt, os.path.join(args.out, "pause_model.pt"))
    json.dump(final, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print("SAVED", os.path.join(args.out, "pause_model.pt"), json.dumps(final))

if __name__ == "__main__":
    main()
