"""Streamlit viewer for the P5 experiment. Read-only: it shows saved results, never runs the pipeline.

Run:  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analysis as A
import config as C
import pandas as pd
import streamlit as st
from common import load_questions

st.set_page_config(page_title="P5: GraphRAG vs naive RAG", layout="wide")


@st.cache_data
def load():
    df = A.load_answers()
    qs = load_questions() if C.QUESTIONS_JSON.exists() else []
    return df, qs


df, qs = load()
page = st.sidebar.radio("Page", ["Overview", "Results (T1, T2, F1)", "Question explorer",
                                 "Example subgraph (F2)", "Failure cases"])
st.sidebar.caption(f"LLM: {C.LLM_MODEL}\n\nEmbedder: {C.EMBED_MODEL}\n\nSeeds: {C.SEEDS}")

if page == "Overview":
    st.title("Naive RAG vs Microsoft GraphRAG on MultiHop-RAG")
    st.write("Three systems answer the same questions with the same small open LLM and embedder. "
             "Metrics: accuracy, indexing time, tokens, graph size.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Questions", len(qs))
    c2.metric("Documents", len(list(C.CORPUS_DIR.glob("*.txt"))) if C.CORPUS_DIR.exists() else 0)
    c3.metric("Seeds", len(C.SEEDS))
    if qs:
        st.subheader("Questions by type")
        st.bar_chart(pd.Series([q["type"] for q in qs]).value_counts())
    if not len(df):
        st.warning("No answer files yet. Run the pipeline scripts (see README) and refresh.")

elif df.empty:
    st.warning("No results yet. Run scripts 02, 04, 05 then 06.")

elif page == "Results (T1, T2, F1)":
    st.subheader("T1: accuracy (mean ± std over seeds, baseline marked)")
    st.dataframe(A.table1(df), hide_index=True, use_container_width=True)
    st.subheader("T2: indexing cost and graph size")
    st.dataframe(A.table2(df), hide_index=True, use_container_width=True)
    st.subheader("F1: accuracy by question type and method")
    st.pyplot(A.figure1(df, C.RESULTS_DIR / "F1_accuracy_by_type.png"))
    st.caption("Accuracy = gold answer found as whole words in the prediction; mean ± std over seeds.")

elif page == "Question explorer":
    ids = [q["id"] for q in qs]
    qid = st.selectbox("Question", ids, format_func=lambda i: f"{i} ({next(q['type'] for q in qs if q['id'] == i)})")
    q = next(x for x in qs if x["id"] == qid)
    st.write(f"**{q['query']}**")
    st.write(f"Gold answer: `{q['answer']}`  ·  type: {q['type']}")
    with st.expander("Gold evidence"):
        for f in q["evidence_facts"]:
            st.write("- " + f)
    seed = st.selectbox("Seed", C.SEEDS)
    cols = st.columns(3)
    for col, m in zip(cols, A.METHODS):
        r = df[(df.method == m) & (df.id == qid) & (df.seed == seed)]
        col.markdown(f"#### {A.LABELS[m]}")
        if r.empty:
            col.info("no result")
            continue
        r = r.iloc[0]
        col.write(("✅ " if r["correct"] else "❌ ") + str(r["pred"])[:600])
        col.caption(f"LLM tokens {int(r['total_llm_tokens']):,} · latency {r['latency_s']:.1f}s")

elif page == "Example subgraph (F2)":
    if "graphrag_local" not in set(df.method):
        st.info("Run GraphRAG local search first.")
    else:
        default = A.pick_example(df, qs)
        ids = [q["id"] for q in qs]
        qid = st.selectbox("Question", ids, index=ids.index(default) if default in ids else 0)
        try:
            st.pyplot(A.figure2(df, qs, qid, C.RESULTS_DIR / "F2_example_subgraph.png"))
        except Exception as e:
            st.error(f"No subgraph stored for this question ({e})")

else:
    st.subheader("Failure cases (seed 0): methods disagree or all fail")
    st.dataframe(A.failure_cases(df, n=10), hide_index=True, use_container_width=True)
    st.caption("Pick 5 for the report's failure analysis and explain the cause (extraction error, "
               "retrieval miss, wrong hop, verbose answer, etc.).")
