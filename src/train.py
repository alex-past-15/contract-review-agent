"""Дообучаем legal-bert на окнах (multi-label). Запуск: python train.py
Учим в fp32: на GTX 1660S с fp16 выходит в 5 раз медленнее."""
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from config import CATEGORIES, DATA, REPORTS, SEED, WINDOWS_PATH

MODEL = "nlpaueb/legal-bert-base-uncased"
MAX_LEN, BATCH, EPOCHS, LR = 384, 8, 3, 3e-5
CKPT = DATA / "legalbert_best"
dev = "cuda"
torch.manual_seed(SEED)


class DS(torch.utils.data.Dataset):
    def __init__(self, df, tok):
        self.enc = tok(list(df.text), truncation=True, max_length=MAX_LEN)
        self.y = df[CATEGORIES].values.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i], "attention_mask": self.enc["attention_mask"][i], "labels": self.y[i]}


def collate(tok):
    def f(batch):
        labels = torch.tensor(np.stack([b.pop("labels") for b in batch]))
        out = tok.pad(batch, return_tensors="pt")
        out["labels"] = labels
        return out
    return f


@torch.no_grad()
def predict(model, dl):
    model.eval()
    ps = []
    for b in dl:
        b = {k: v.to(dev) for k, v in b.items() if k != "labels"}
        ps.append(torch.sigmoid(model(**b).logits).cpu().numpy())
    return np.concatenate(ps)


def macro_f1(y, p, thr=0.5):
    return np.mean([precision_recall_fscore_support(y[:, i], p[:, i] >= thr, average="binary", zero_division=0)[2]
                    for i in range(y.shape[1])])


def main():
    df = pd.read_parquet(WINDOWS_PATH)
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    tok = AutoTokenizer.from_pretrained(MODEL)
    col = collate(tok)
    dl_tr = DataLoader(DS(tr, tok), batch_size=BATCH, shuffle=True, collate_fn=col)
    dl_va = DataLoader(DS(va, tok), batch_size=32, collate_fn=col)
    dl_te = DataLoader(DS(te, tok), batch_size=32, collate_fn=col)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(CATEGORIES), problem_type="multi_label_classification").to(dev)
    pos = tr[CATEGORIES].values.mean(0)
    pos_weight = torch.tensor(np.clip((1 - pos) / pos, 1, 10) ** 0.5, dtype=torch.float32, device=dev)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * len(dl_tr) * EPOCHS), len(dl_tr) * EPOCHS)

    best, yva = -1, va[CATEGORIES].values
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for i, b in enumerate(dl_tr):
            b = {k: v.to(dev) for k, v in b.items()}
            labels = b.pop("labels")
            loss = loss_fn(model(**b).logits, labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if i % 100 == 0:
                print(f"ep{ep} step {i}/{len(dl_tr)} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
        f1 = macro_f1(yva, predict(model, dl_va))
        print(f"== epoch {ep} val macro-F1@0.5 = {f1:.4f}", flush=True)
        if f1 > best:
            best = f1
            model.save_pretrained(CKPT)
            tok.save_pretrained(CKPT)

    # оцениваем лучший чекпойнт, пороги берём по val
    model = AutoModelForSequenceClassification.from_pretrained(CKPT).to(dev)
    pva, pte, yte = predict(model, dl_va), predict(model, dl_te), te[CATEGORIES].values
    np.save(DATA / "test_probs_legalbert.npy", pte)
    rows = {}
    for i, cat in enumerate(CATEGORIES):
        t = max(np.arange(0.1, 0.95, 0.05),
                key=lambda t: precision_recall_fscore_support(yva[:, i], pva[:, i] >= t, average="binary", zero_division=0)[2])
        p, r, f, _ = precision_recall_fscore_support(yte[:, i], pte[:, i] >= t, average="binary", zero_division=0)
        rows[cat] = {"precision": p, "recall": r, "f1": f, "threshold": float(t), "test_positives": int(yte[:, i].sum())}
    res = pd.DataFrame(rows).T
    res.loc["MACRO"] = res[["precision", "recall", "f1"]].mean().tolist() + [np.nan, res.test_positives.sum()]
    print(res.round(3).to_string())
    res.round(4).to_csv(REPORTS / "legalbert.csv")


if __name__ == "__main__":
    main()
