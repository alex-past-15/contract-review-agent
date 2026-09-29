"""Считаем метрики агента по договорам и экономию. Данные берём из reports/agent_eval.jsonl."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from config import CATEGORIES, CUAD_JSON, REPORTS
from roi import Assumptions, monthly_effect


def main():
    recs = [json.loads(line) for line in open(REPORTS / "agent_eval.jsonl", encoding="utf-8")]
    truth = {}
    for c in json.load(open(CUAD_JSON, encoding="utf-8"))["data"]:
        truth[c["title"]] = {q["id"].split("__")[-1] for q in c["paragraphs"][0]["qas"]
                             if not q["is_impossible"]} & set(CATEGORIES)

    rows = {}
    for cat in CATEGORIES:
        y = [cat in truth[r["contract"]] for r in recs]
        p = [cat in r["found"] for r in recs]
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
        rows[cat] = {"precision": pr, "recall": rc, "f1": f1, "contracts_with_clause": int(sum(y))}
    res = pd.DataFrame(rows).T
    micro_y = np.array([[c in truth[r["contract"]] for c in CATEGORIES] for r in recs]).ravel()
    micro_p = np.array([[c in r["found"] for c in CATEGORIES] for r in recs]).ravel()
    mp, mr, mf, _ = precision_recall_fscore_support(micro_y, micro_p, average="binary", zero_division=0)
    res.loc["MACRO"] = res[["precision", "recall", "f1"]].mean().tolist() + [res.contracts_with_clause.sum()]
    res.loc["MICRO"] = [mp, mr, mf, res.contracts_with_clause.iloc[:-1].sum()]
    res.round(3).to_csv(REPORTS / "agent_contract_level.csv")

    sec = np.array([r["seconds"] for r in recs])
    grounded = [v for r in recs for v in r["grounded"].values() if v is not None]
    n_items = sum(len(r["found"]) for r in recs)
    n_failed_final = sum(len(r["failed"]) for r in recs)
    first_try = np.mean([r["retries"] == 0 for r in recs])
    summary = {
        "contracts": len(recs),
        "micro_precision": round(mp, 3), "micro_recall": round(mr, 3), "micro_f1": round(mf, 3),
        "macro_f1": round(res.loc["MACRO", "f1"], 3),
        "sec_per_contract_mean": round(sec.mean(), 1), "sec_per_contract_median": round(float(np.median(sec)), 1),
        "contracts_needing_human": int(sum(r["needs_human"] for r in recs)),
        "share_no_retry": round(float(first_try), 3),
        "clauses_found_total": n_items,
        "clauses_unverified_after_retries": n_failed_final,
        "share_summaries_verified": round(1 - n_failed_final / max(n_items, 1), 3),
        "mean_quote_grounding": round(float(np.mean(grounded)), 3) if grounded else None,
    }
    roi = {"assumptions": Assumptions().__dict__, "effect": monthly_effect(Assumptions(), mr)}
    (REPORTS / "agent_summary.json").write_text(json.dumps({"summary": summary, "roi": roi}, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print(res.round(3).to_string())
    print(json.dumps({"summary": summary, "roi": roi}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
