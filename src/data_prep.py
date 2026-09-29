"""Режем договоры CUAD на окна и размечаем их по категориям, сплит делаем по договорам."""
import json
import random

import pandas as pd

from config import CATEGORIES, CUAD_JSON, SEED, STRIDE_CHARS, WINDOW_CHARS, WINDOWS_PATH


def split_contracts(titles, seed=SEED):
    titles = sorted(titles)
    random.Random(seed).shuffle(titles)
    n = len(titles)
    n_train, n_val = int(n * 0.7), int(n * 0.15)
    split = {}
    for i, t in enumerate(titles):
        split[t] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return split


def make_windows(text, spans):
    """spans: {категория: [(start, end), ...]}. Окно получает метку, если покрывает хотя бы половину пункта."""
    rows = []
    for start in range(0, max(len(text) - WINDOW_CHARS + STRIDE_CHARS, 1), STRIDE_CHARS):
        end = min(start + WINDOW_CHARS, len(text))
        labels = {}
        for cat in CATEGORIES:
            hit = 0
            for s, e in spans.get(cat, []):
                overlap = min(end, e) - max(start, s)
                if overlap >= 0.5 * min(e - s, WINDOW_CHARS):
                    hit = 1
                    break
            labels[cat] = hit
        rows.append({"start": start, "end": end, "text": text[start:end], **labels})
        if end == len(text):
            break
    return rows


def main():
    data = json.load(open(CUAD_JSON, encoding="utf-8"))["data"]
    split = split_contracts([c["title"] for c in data])
    rows = []
    for c in data:
        para = c["paragraphs"][0]
        text = para["context"]
        spans = {}
        for q in para["qas"]:
            cat = q["id"].split("__")[-1]
            if cat in CATEGORIES:
                spans[cat] = [(a["answer_start"], a["answer_start"] + len(a["text"])) for a in q["answers"]]
        for w in make_windows(text, spans):
            rows.append({"contract": c["title"], "split": split[c["title"]], **w})
    df = pd.DataFrame(rows)
    df.to_parquet(WINDOWS_PATH)
    print(df.groupby("split").size())
    print(df.groupby("split")[CATEGORIES].sum().T)


if __name__ == "__main__":
    main()
