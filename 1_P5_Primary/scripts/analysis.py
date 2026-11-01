"""Scoring, tables (T1, T2) and figures (F1, F2). Used by 06_evaluate.py and the Streamlit app."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
from common import contains_match, exact_match, f1, load_questions, read_jsonl, normalize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["naive", "graphrag_local", "graphrag_global"]
LABELS = {"naive": "Naive RAG (baseline)", "graphrag_local": "GraphRAG local", "graphrag_global": "GraphRAG global"}
COLORS = {"naive": "#1D9E75", "graphrag_local": "#534AB7", "graphrag_global": "#AFA9EC"}
TYPES = ["inference", "comparison", "temporal"]


def load_answers() -> pd.DataFrame:
    frames = []
    for p in (C.NAIVE_ANSWERS, C.LOCAL_ANSWERS, C.GLOBAL_ANSWERS):
        if Path(p).exists():
            frames.append(pd.DataFrame(read_jsonl(p)))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["correct"] = [contains_match(p, g) for p, g in zip(df["pred"], df["gold"])]
    df["em"] = [exact_match(p, g) for p, g in zip(df["pred"], df["gold"])]
    df["f1"] = [f1(p, g) for p, g in zip(df["pred"], df["gold"])]
    for c in ("prompt_tokens", "completion_tokens", "embedding_tokens"):
        df[c] = df[c].fillna(0)
    df["total_llm_tokens"] = df["prompt_tokens"] + df["completion_tokens"]
    return df


def _ms(x):
    return f"{np.mean(x) * 100:.1f} ± {np.std(x, ddof=1) * 100:.1f}" if len(x) > 1 else f"{np.mean(x) * 100:.1f}"


def per_seed_accuracy(df, by_type=False):
    keys = ["method", "seed"] + (["type"] if by_type else [])
    return df.groupby(keys)["correct"].mean().reset_index()


def table1(df) -> pd.DataFrame:
    """T1: accuracy (%) mean ± std over seeds, overall and per question type."""
    rows = []
    overall = per_seed_accuracy(df)
    bytype = per_seed_accuracy(df, by_type=True)
    em = df.groupby(["method", "seed"])["em"].mean().reset_index()
    f = df.groupby(["method", "seed"])["f1"].mean().reset_index()
    for m in METHODS:
        if m not in set(df["method"]):
            continue
        r = {"Method": LABELS[m] + (" [baseline]" if m == "naive" else ""),
             "Accuracy (%)": _ms(overall[overall.method == m]["correct"]),
             "EM (%)": _ms(em[em.method == m]["em"]),
             "F1 (%)": _ms(f[f.method == m]["f1"])}
        for t in TYPES:
            r[f"{t.capitalize()} (%)"] = _ms(bytype[(bytype.method == m) & (bytype.type == t)]["correct"])
        rows.append(r)
    return pd.DataFrame(rows)


def table2(df) -> pd.DataFrame:
    """T2: indexing time, tokens, graph size, plus query-time cost."""
    st = json.loads(C.INDEX_STATS.read_text()) if C.INDEX_STATS.exists() else {}
    rows = []
    for m in METHODS:
        key = "graphrag" if m.startswith("graphrag") else "naive"
        s = st.get(key, {})
        q = df[df.method == m] if len(df) else pd.DataFrame()
        shared = " (shared index)" if m.startswith("graphrag") else ""
        rows.append({
            "Method": LABELS[m],
            "Indexing time (s)": f"{s['indexing_seconds']:.0f}{shared}" if s else "n/a",
            "Index LLM tokens": (f"{s['llm_prompt_tokens'] + s['llm_completion_tokens']:,}" if s else "n/a"),
            "Index embedding tokens": f"{s['embedding_tokens']:,}" if s else "n/a",
            "Nodes": (s.get("num_nodes") if s.get("num_nodes") is not None else "–") if s else "n/a",
            "Edges": (s.get("num_edges") if s.get("num_edges") is not None else "–") if s else "n/a",
            "Chunks": s.get("num_chunks", "n/a") if s else "n/a",
            "Query LLM tokens / question": f"{q['total_llm_tokens'].mean():,.0f}" if len(q) else "n/a",
            "Latency / question (s)": f"{q['latency_s'].mean():.1f}" if len(q) else "n/a"})
    return pd.DataFrame(rows)


def results_long(df) -> pd.DataFrame:
    """results.csv: one row per (method, question_type, seed, metric)."""
    rows = []
    for (m, s), g in df.groupby(["method", "seed"]):
        groups = [("all", g)] + [(t, g[g.type == t]) for t in TYPES]
        for t, gg in groups:
            for metric, val in (("accuracy", gg.correct.mean()), ("em", gg.em.mean()), ("f1", gg.f1.mean()),
                                ("llm_tokens_per_question", gg.total_llm_tokens.mean()),
                                ("latency_s_per_question", gg.latency_s.mean())):
                rows.append({"method": m, "question_type": t, "seed": s, "metric": metric,
                             "value": float(val), "n_questions": len(gg)})
    st = json.loads(C.INDEX_STATS.read_text()) if C.INDEX_STATS.exists() else {}
    for key, s in st.items():
        for metric, val in s.items():
            if val is not None:
                rows.append({"method": f"index_{key}", "question_type": "all", "seed": "", "metric": metric,
                             "value": float(val), "n_questions": ""})
    return pd.DataFrame(rows)


def figure1(df, path):
    """F1: accuracy by question type and method (mean ± std over seeds)."""
    bt = per_seed_accuracy(df, by_type=True)
    ov = per_seed_accuracy(df)
    ov["type"] = "overall"
    d = pd.concat([bt, ov])
    cats = TYPES + ["overall"]
    ms = [m for m in METHODS if m in set(df.method)]
    w = 0.8 / len(ms)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, m in enumerate(ms):
        means = [d[(d.method == m) & (d.type == c)]["correct"].mean() * 100 for c in cats]
        stds = [d[(d.method == m) & (d.type == c)]["correct"].std(ddof=1) * 100 if
                (d[(d.method == m) & (d.type == c)]["correct"].count() > 1) else 0 for c in cats]
        ax.bar(np.arange(len(cats)) + i * w - 0.4 + w / 2, means, w, yerr=stds, capsize=3,
               label=LABELS[m], color=COLORS[m])
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c.capitalize() for c in cats])
    ax.set_xlabel("Question type")
    ax.set_ylabel("Accuracy (%, mean ± std over seeds)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return fig


def pick_example(df, qs):
    """Prefer a question GraphRAG-local answered correctly with a mid-sized retrieved subgraph."""
    loc = df[(df.method == "graphrag_local") & (df.seed == C.SEEDS[0])]
    best = None
    for _, r in loc.iterrows():
        sg = r.get("subgraph")
        if not isinstance(sg, dict) or not sg.get("relationships"):
            continue
        n = len(sg["entities"])
        score = (bool(r["correct"]), 8 <= n <= 40, -abs(n - 20))
        if best is None or score > best[0]:
            best = (score, r["id"])
    return best[1] if best else None


def figure2(df, qs, qid, path, max_nodes=30):
    """F2: the entity/relationship subgraph GraphRAG-local put into the prompt for one question."""
    q = next(x for x in qs if x["id"] == qid)
    r = df[(df.method == "graphrag_local") & (df.id == qid)].iloc[0]
    sg = r["subgraph"]
    G = nx.Graph()
    for rel in sg["relationships"]:
        G.add_edge(rel["source"], rel["target"])
    for e in sg["entities"]:
        G.add_node(e)
    if G.number_of_nodes() > max_nodes:
        keep = [n for n, _ in sorted(G.degree, key=lambda x: -x[1])[:max_nodes]]
        G = G.subgraph(keep).copy()
    gold_text = normalize(q["answer"] + " " + " ".join(q["evidence_facts"]))
    hit = {n for n in G if normalize(n) and normalize(n) in gold_text}
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=C.SUBSET_SEED, k=1.2 / max(np.sqrt(len(G)), 1))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, width=1)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[520 if n in hit else 260 for n in G],
                           node_color=["#D85A30" if n in hit else "#AFA9EC" for n in G], linewidths=0)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
    ax.set_title(f"{q['id']} ({q['type']}): {q['query'][:90]}…", fontsize=8, loc="left")
    ax.text(0, -0.02, f"Gold answer: {q['answer']}  |  GraphRAG-local answer: {r['pred'][:80]}  |  "
            f"orange = entity appears in gold evidence", transform=ax.transAxes, fontsize=7, va="top")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return fig


def failure_cases(df, n=5):
    """Cases where the baseline and at least one GraphRAG mode disagree or all fail (seed 0)."""
    d = df[df.seed == C.SEEDS[0]].pivot_table(index=["id", "type", "gold"], columns="method",
                                              values="correct", aggfunc="first").reset_index()
    preds = df[df.seed == C.SEEDS[0]].pivot_table(index="id", columns="method", values="pred", aggfunc="first")
    d = d.merge(preds.add_prefix("pred_"), left_on="id", right_index=True)
    meth = [m for m in METHODS if m in d.columns]
    d["n_correct"] = d[meth].sum(axis=1)
    d["disagree"] = d[meth].nunique(axis=1) > 1
    d = d.sort_values(["disagree", "n_correct"], ascending=[False, True])
    return d.head(n)
