"""Streamlit-демо. Запуск: streamlit run src/app.py"""
import json
import time

import streamlit as st

from config import CUAD_JSON, REPORTS
from pipeline import build_graph
from roi import Assumptions, monthly_effect

st.set_page_config(page_title="Contract Review Agent", layout="centered")
st.title("Contract Review Agent")
st.caption("Проверка договора: найдёт рискованные пункты, кратко объяснит и приведёт цитаты. "
           "Всё работает локально, без облачных API.")


@st.cache_resource
def graph():
    return build_graph()


@st.cache_data
def examples():
    return {c["title"]: c["paragraphs"][0]["context"]
            for c in json.load(open(CUAD_JSON, encoding="utf-8"))["data"]}


docs = examples()
mode = st.radio("Что проверяем?", ["Пример договора", "Свой договор"], horizontal=True)
if mode == "Пример договора":
    text = docs[st.selectbox("Договор", list(docs))]
    st.caption(f"{len(text):,} символов".replace(",", " "))
else:
    up = st.file_uploader("Загрузить файл .txt", type="txt")
    pasted = st.text_area("или вставить текст договора (на английском)", height=220)
    text = up.read().decode("utf-8", errors="ignore") if up else pasted
    if text.strip():
        st.caption(f"{len(text):,} символов".replace(",", " "))

if st.button("Проверить договор", type="primary", disabled=not text.strip()):
    t0 = time.time()
    with st.spinner("Анализирую договор, это займёт около минуты…"):
        out = graph().invoke({"text": text})
    st.success(f"Готово за {time.time() - t0:.0f} с")
    if out.get("needs_human"):
        st.warning("Часть выводов не подтверждена цитатами. Нужна проверка юристом.")
    st.markdown(out["report"])

with st.expander("Качество и экономика проекта"):
    s = REPORTS / "agent_summary.json"
    if s.exists():
        m = json.loads(s.read_text(encoding="utf-8"))["summary"]
        c = st.columns(3)
        c[0].metric("F1 по договорам", m["micro_f1"])
        c[1].metric("Найдено пунктов (recall)", m["micro_recall"])
        c[2].metric("Секунд на договор", m["sec_per_contract_mean"])
        e = monthly_effect(Assumptions(), m["micro_recall"])
        saving = f"{e['net_saving']:,}".replace(",", " ")
        st.write(f"Ориентировочная экономия: **~{saving} руб. в месяц** и ~{e['hours_saved']:.0f} ч работы юриста "
                 "при 200 договорах в месяц.")
        st.caption("Метрики измерены на 77 тестовых договорах. Экономика - это расчёт с допущениями "
                   "(см. src/roi.py и README), а не результат внедрения.")
    else:
        st.info("Метрики появятся после запуска eval_agent.py и metrics.py.")
