"""Пайплайн на LangGraph: segment -> classify -> score_risk -> summarize <-> critic -> report."""
import difflib
import json
import re
from typing import TypedDict

import numpy as np
import pandas as pd
import torch
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import risk
from config import CATEGORIES, DATA, REPORTS, STRIDE_CHARS, WINDOW_CHARS

LLM_MODEL = "qwen2.5:7b"
MAX_RETRIES = 2
QUOTE_MATCH = 0.9  # какая доля цитаты должна найтись в договоре


class State(TypedDict, total=False):
    text: str
    windows: list[dict]
    evidence: dict          # категория: {"text", "prob", "start"}
    risks: list[dict]
    summaries: dict         # категория: {"summary", "quote", "grounded"}
    failed: dict            # категория: почему критик не принял выжимку
    retries: int
    needs_human: bool
    report: str


# классификатор грузим только когда он нужен
_clf = {}


def _load_clf():
    if not _clf:
        tok = AutoTokenizer.from_pretrained(DATA / "legalbert_best")
        model = AutoModelForSequenceClassification.from_pretrained(DATA / "legalbert_best")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _clf.update(tok=tok, model=model.to(dev).eval(), dev=dev,
                    thr=pd.read_csv(REPORTS / "legalbert.csv", index_col=0)["threshold"])
    return _clf


def free_classifier():
    """Освобождаем видеопамять перед вызовом LLM, иначе на 6 GB не влезет."""
    _clf.clear()
    torch.cuda.empty_cache()


def segment(state: State) -> State:
    text, wins = state["text"], []
    for start in range(0, max(len(text) - WINDOW_CHARS + STRIDE_CHARS, 1), STRIDE_CHARS):
        end = min(start + WINDOW_CHARS, len(text))
        wins.append({"start": start, "text": text[start:end]})
        if end == len(text):
            break
    return {"windows": wins, "retries": 0, "failed": {}, "summaries": {}}


@torch.no_grad()
def classify(state: State) -> State:
    c = _load_clf()
    texts = [w["text"] for w in state["windows"]]
    probs = []
    for i in range(0, len(texts), 32):
        enc = c["tok"](texts[i:i + 32], truncation=True, max_length=384, padding=True,
                       return_tensors="pt").to(c["dev"])
        probs.append(torch.sigmoid(c["model"](**enc).logits).cpu().numpy())
    probs = np.concatenate(probs)
    evidence = {}
    for j, cat in enumerate(CATEGORIES):
        best = int(probs[:, j].argmax())
        if probs[best, j] >= c["thr"][cat]:
            evidence[cat] = {"text": texts[best], "prob": float(probs[best, j]),
                             "start": state["windows"][best]["start"]}
    free_classifier()
    return {"evidence": evidence}


def score_risk(state: State) -> State:
    return {"risks": risk.score(set(state["evidence"]), CATEGORIES)}


PROMPT = '''Ты помощник юриста. Ниже фрагмент договора, который относится к теме «{cat}».
Задача:
1. Кратко (1-2 предложения, по-русски) опиши, что именно закреплено в этом фрагменте.
2. Дай ДОСЛОВНУЮ цитату из фрагмента (на языке оригинала, до 250 символов), подтверждающую твоё описание.
Не добавляй ничего, чего нет во фрагменте. Если во фрагменте нет пункта по теме, верни summary "NOT_FOUND" и пустую цитату.
{feedback}
Фрагмент:
"""{text}"""

Ответ строго в JSON: {{"summary": "...", "quote": "..."}}'''

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOllama(model=LLM_MODEL, temperature=0, format="json", num_ctx=4096,
                           client_kwargs={"timeout": 120})
    return _llm


def summarize(state: State) -> State:
    todo = list(state["failed"]) or list(state["evidence"])
    summaries = dict(state["summaries"])
    for cat in todo:
        fb = ""
        if cat in state["failed"]:
            fb = f"ВНИМАНИЕ: прошлая цитата не найдена в тексте ({state['failed'][cat]}). Копируй цитату символ в символ.\n"
        prompt = PROMPT.format(cat=cat, text=state["evidence"][cat]["text"], feedback=fb)
        try:
            out = json.loads(_get_llm().invoke(prompt).content)
        except Exception:
            out = {"summary": "", "quote": ""}
        summaries[cat] = {"summary": str(out.get("summary", "")).strip(),
                          "quote": str(out.get("quote", "")).strip()}
    return {"summaries": summaries}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def quote_supported(quote: str, source: str) -> float:
    """Какая доля цитаты нашлась в источнике (1.0 - цитата дословная)."""
    q, s = _norm(quote), _norm(source)
    if not q:
        return 0.0
    if q in s:
        return 1.0
    blocks = difflib.SequenceMatcher(None, s, q, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks) / len(q)


def critic(state: State) -> State:
    failed = {}
    for cat, item in state["summaries"].items():
        if item["summary"] == "NOT_FOUND":
            failed[cat] = "модель не нашла пункт во фрагменте"
            continue
        score = quote_supported(item["quote"], state["evidence"][cat]["text"])
        item["grounded"] = round(score, 3)
        if score < QUOTE_MATCH:
            failed[cat] = f"совпадение цитаты {score:.0%}"
    retries = state["retries"] + (1 if failed else 0)
    return {"failed": failed, "retries": retries, "needs_human": bool(failed) and retries > MAX_RETRIES}


LEVEL = {"high": "🔴 высокий", "medium": "🟡 средний", "low": "🟢 низкий"}


def report(state: State) -> State:
    risks = state.get("risks", [])
    lines = ["# Отчёт по договору\n",
             f"**Общий риск:** {LEVEL[risk.overall(risks)] if risks else 'н/д'}\n"]
    if state.get("needs_human"):
        lines.append("> ⚠️ Часть выводов не подтверждена цитатами. Требуется проверка юристом.\n")
    for r in risks:
        lines.append(f"## {LEVEL[r['risk']]} - {r['category']} ({'найден' if r['status'] == 'found' else 'не найден'})")
        lines.append(f"{r['why']}")
        s = state["summaries"].get(r["category"])
        if s and r["category"] not in state["failed"]:
            lines.append(f"\n**Суть:** {s['summary']}\n\n> {s['quote']}\n")
        elif r["category"] in state["failed"]:
            lines.append("\n_Автоматическая выжимка не подтверждена, смотрите оригинал._\n")
    return {"report": "\n".join(lines)}


def after_classify(state: State) -> str:
    return "score_risk" if state["evidence"] else "report"


def after_score(state: State) -> str:
    return "summarize" if state["evidence"] else "report"


def after_critic(state: State) -> str:
    return "summarize" if state["failed"] and state["retries"] <= MAX_RETRIES else "report"


def build_graph():
    g = StateGraph(State)
    for name, fn in [("segment", segment), ("classify", classify), ("score_risk", score_risk),
                     ("summarize", summarize), ("critic", critic), ("report", report)]:
        g.add_node(name, fn)
    g.add_edge(START, "segment")
    g.add_edge("segment", "classify")
    g.add_conditional_edges("classify", after_classify, ["score_risk", "report"])
    g.add_conditional_edges("score_risk", after_score, ["summarize", "report"])
    g.add_edge("summarize", "critic")
    g.add_conditional_edges("critic", after_critic, ["summarize", "report"])
    g.add_edge("report", END)
    return g.compile()
