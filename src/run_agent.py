"""Пример: python run_agent.py [часть названия договора из CUAD | путь к .txt]"""
import json
import sys
import time
from pathlib import Path

from config import CUAD_JSON, REPORTS
from pipeline import build_graph


def load_text(arg):
    if arg and Path(arg).exists():
        return Path(arg).read_text(encoding="utf-8"), Path(arg).stem
    data = json.load(open(CUAD_JSON, encoding="utf-8"))["data"]
    c = next((c for c in data if arg and arg in c["title"]), data[0])
    return c["paragraphs"][0]["context"], c["title"]


if __name__ == "__main__":
    text, name = load_text(sys.argv[1] if len(sys.argv) > 1 else None)
    t0 = time.time()
    out = build_graph().invoke({"text": text})
    print(out["report"])
    print(f"\n[{name}] {len(text)} симв., {time.time() - t0:.0f} c, retries={out['retries']}, "
          f"needs_human={out.get('needs_human')}")
    (REPORTS / "last_report.md").write_text(out["report"], encoding="utf-8")
