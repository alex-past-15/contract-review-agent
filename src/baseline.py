"""Baseline: TF-IDF + LogisticRegression (one-vs-rest) по окнам."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

from config import CATEGORIES, REPORTS, WINDOWS_PATH


def main():
    df = pd.read_parquet(WINDOWS_PATH)
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)
    Xtr = vec.fit_transform(tr.text)
    Xva, Xte = vec.transform(va.text), vec.transform(te.text)

    report = {}
    for cat in CATEGORIES:
        clf = LogisticRegression(C=5, class_weight="balanced", max_iter=1000)
        clf.fit(Xtr, tr[cat])
        # порог выбираем по F1 на val
        pv = clf.predict_proba(Xva)[:, 1]
        best_t = max(np.arange(0.1, 0.95, 0.05),
                     key=lambda t: precision_recall_fscore_support(va[cat], pv >= t, average="binary", zero_division=0)[2])
        pt = clf.predict_proba(Xte)[:, 1]
        p, r, f, _ = precision_recall_fscore_support(te[cat], pt >= best_t, average="binary", zero_division=0)
        report[cat] = {"precision": p, "recall": r, "f1": f, "threshold": float(best_t), "test_positives": int(te[cat].sum())}

    res = pd.DataFrame(report).T
    res.loc["MACRO"] = res[["precision", "recall", "f1"]].mean().tolist() + [np.nan, res.test_positives.sum()]
    print(res.round(3).to_string())
    REPORTS.mkdir(exist_ok=True)
    res.round(4).to_csv(REPORTS / "baseline_tfidf.csv")


if __name__ == "__main__":
    main()
