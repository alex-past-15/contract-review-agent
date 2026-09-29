"""Прогоняем агента по тестовым договорам. Результат пишется в reports/agent_eval.jsonl,
при перезапуске уже обработанные договоры пропускаются."""
import json
import time

import pandas as pd

from config import CUAD_JSON, DATA, REPORTS
from pipeline import build_graph

OUT = REPORTS / "agent_eval.jsonl"


def main():
    test_titles = set(pd.read_parquet(DATA / "windows.parquet", columns=["contract", "split"])
                      .query("split == 'test'").contract)
    data = [c for c in json.load(open(CUAD_JSON, encoding="utf-8"))["data"] if c["title"] in test_titles]
    done = set()
    if OUT.exists():
        done = {json.loads(line)["contract"] for line in open(OUT, encoding="utf-8")}
    graph = build_graph()
    print(f"test contracts: {len(data)}, already done: {len(done)}", flush=True)
    for i, c in enumerate(data):
        if c["title"] in done:
            continue
        text = c["paragraphs"][0]["context"]
        t0 = time.time()
        out = graph.invoke({"text": text})
        rec = {
            "contract": c["title"],
            "chars": len(text),
            "seconds": round(time.time() - t0, 1),
            "found": sorted(out["evidence"]),
            "retries": out["retries"],
            "needs_human": bool(out.get("needs_human")),
            "failed": sorted(out["failed"]),
            "grounded": {k: v.get("grounded") for k, v in out["summaries"].items()},
            "overall_risk": max((r["risk"] for r in out["risks"]), key=["low", "medium", "high"].index) if out.get("risks") else None,
        }
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{i + 1}/{len(data)}] {c['title'][:50]} {rec['seconds']}s found={len(rec['found'])} retries={rec['retries']}", flush=True)


if __name__ == "__main__":
    main()
