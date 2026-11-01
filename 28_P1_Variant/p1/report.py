"""Build the report tables (T1, T2) and figures (F1, F2) from results/:  python -m p1.report

Outputs in results/: T1_quality.csv, T2_cost.csv, T3_ndcg_ci.csv, tables.md,
F1_ndcg_by_strategy.png, F2_recall_curves.png, F3_hotpot_by_qtype.png,
failure_cases.csv, doc_type_summary.csv
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .metrics import bootstrap_ci

COL = {"fixed": "#1D9E75", "recursive": "#534AB7", "semantic": "#D85A30", "whole_doc": "#888780"}
NAME = {"fixed": "Fixed-length", "recursive": "Recursive", "semantic": "Semantic", "whole_doc": "No chunking (baseline)"}
DS_NAME = {"nfcorpus": "NFCorpus", "hotpotqa": "HotpotQA subset"}


def _ms(x):
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}" if len(x) > 1 else f"{x.mean():.3f}"


def _label(s, z):
    return NAME[s] if s == C.BASELINE else f"{NAME[s]} ({z})"


def _order(df):
    keys = [(C.BASELINE, 0)] + [(s, z) for s in C.STRATEGIES for z in C.CHUNK_SIZES]
    return [k for k in keys if ((df.strategy == k[0]) & (df.chunk_size == k[1])).any()]


def table1(res):
    rows = []
    for ds, g in res.groupby("dataset"):
        for s, z in _order(g):
            sub = g[(g.strategy == s) & (g.chunk_size == z)]
            row = {"Dataset": DS_NAME[ds], "Strategy": _label(s, z) + (" [baseline]" if s == C.BASELINE else "")}
            for m, lab in (("recall@5", "Recall@5"), ("recall@10", "Recall@10"), ("mrr@10", "MRR@10"), ("ndcg@10", "NDCG@10")):
                row[lab] = _ms(sub[sub.metric == m].sort_values("seed")["value"])
            rows.append(row)
    return pd.DataFrame(rows)


def table2(cost):
    rows = []
    for ds, g in cost.groupby("dataset"):
        for s, z in _order(g):
            sub = g[(g.strategy == s) & (g.chunk_size == z)]
            rows.append({"Dataset": DS_NAME[ds], "Strategy": _label(s, z) + (" [baseline]" if s == C.BASELINE else ""),
                         "Chunks": f"{sub.n_chunks.mean():,.0f}", "Mean tokens/chunk": f"{sub.mean_chunk_tokens.mean():.0f}",
                         "Chunking time (s)": f"{sub.chunk_seconds.mean():.2f}",
                         "Embedding time (s)": f"{sub.embed_seconds.mean():.2f}",
                         "Indexing time (s)": f"{sub.index_seconds.mean():.2f}"})
    return pd.DataFrame(rows)


def table3_ci(pq):
    rows = []
    for ds, g in pq.groupby("dataset"):
        for s, z in _order(g):
            v = g[(g.strategy == s) & (g.chunk_size == z)]["ndcg@10"]
            lo, hi = bootstrap_ci(v, C.BOOTSTRAP)
            rows.append({"Dataset": DS_NAME[ds], "Strategy": _label(s, z), "NDCG@10": f"{v.mean():.3f}",
                         "95% bootstrap CI": f"[{lo:.3f}, {hi:.3f}]", "Queries (pooled over seeds)": len(v)})
    return pd.DataFrame(rows)


def figure1(res, path):
    dsets = list(res.dataset.unique())
    fig, axes = plt.subplots(1, len(dsets), figsize=(5.5 * len(dsets), 4), squeeze=False)
    for ax, ds in zip(axes[0], dsets):
        g = res[(res.dataset == ds) & (res.metric == "ndcg@10")]
        w = 0.8 / len(C.CHUNK_SIZES)
        for i, z in enumerate(C.CHUNK_SIZES):
            vals, errs = [], []
            for s in C.STRATEGIES:
                v = g[(g.strategy == s) & (g.chunk_size == z)]["value"]
                vals.append(v.mean()); errs.append(v.std(ddof=1) if len(v) > 1 else 0)
            ax.bar(np.arange(len(C.STRATEGIES)) + i * w - 0.4 + w / 2, vals, w, yerr=errs, capsize=3,
                   color=[COL[s] for s in C.STRATEGIES], alpha=0.55 if i == 0 else 1.0,
                   edgecolor="white", label=f"{z} tokens")
        b = g[g.strategy == C.BASELINE]["value"].mean()
        ax.axhline(b, color=COL["whole_doc"], ls="--", lw=1, label="No chunking (baseline)")
        ax.set_xticks(range(len(C.STRATEGIES)))
        ax.set_xticklabels([NAME[s] for s in C.STRATEGIES])
        ax.set_xlabel("Chunking strategy (light = %d tokens, dark = %d tokens)" % tuple(C.CHUNK_SIZES))
        ax.set_ylabel("NDCG@10 (mean ± std over seeds)")
        ax.set_title(DS_NAME[ds])
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def figure2(res, path):
    dsets = list(res.dataset.unique())
    fig, axes = plt.subplots(1, len(dsets), figsize=(5.5 * len(dsets), 4), squeeze=False)
    for ax, ds in zip(axes[0], dsets):
        g = res[res.dataset == ds]
        for s, z in _order(g):
            sub = g[(g.strategy == s) & (g.chunk_size == z) & g.metric.str.startswith("recall@")].copy()
            sub["k"] = sub.metric.str.split("@").str[1].astype(int)
            c = sub.groupby("k")["value"].mean()
            ax.plot(c.index, c.values, color=COL[s], lw=1.6, ls="-" if (s == C.BASELINE or z == C.CHUNK_SIZES[-1]) else "--",
                    label=_label(s, z))
        ax.set_xlabel("Cutoff K (number of retrieved documents)")
        ax.set_ylabel("Recall@K")
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_title(DS_NAME[ds])
        ax.legend(frameon=False, fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def figure3_hotpot(pq, path):
    g = pq[(pq.dataset == "hotpotqa") & (pq.qtype != "")]
    if g.empty:
        return
    types = sorted(g.qtype.unique())
    keys = _order(g)
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.8 / len(types)
    for i, t in enumerate(types):
        vals = [g[(g.qtype == t) & (g.strategy == s) & (g.chunk_size == z)]["ndcg@10"].mean() for s, z in keys]
        ax.bar(np.arange(len(keys)) + i * w - 0.4 + w / 2, vals, w, label=t)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([_label(s, z).replace(" (", "\n(") for s, z in keys], fontsize=7)
    ax.set_xlabel("Strategy (chunk size in tokens)")
    ax.set_ylabel("NDCG@10 (mean over seeds)")
    ax.legend(title="HotpotQA question type", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def failure_cases(pq, n=5):
    """Queries (seed 0) where the best and worst chunking configuration differ most in NDCG@10."""
    rows = []
    for ds, g in pq[pq.seed == 0].groupby("dataset"):
        wide = g.assign(cfg=g.strategy + "_" + g.chunk_size.astype(str)).pivot(index="qid", columns="cfg", values="ndcg@10")
        gap = (wide.max(axis=1) - wide.min(axis=1)).sort_values(ascending=False)
        for qid in gap.index[: (3 if ds == "nfcorpus" else 2)]:
            best, worst = wide.loc[qid].idxmax(), wide.loc[qid].idxmin()
            sel = g[g.qid == qid]
            get = lambda c: sel[(sel.strategy + "_" + sel.chunk_size.astype(str)) == c].iloc[0]
            rows.append({"dataset": ds, "qid": qid, "best_config": best, "best_ndcg@10": wide.loc[qid, best],
                         "worst_config": worst, "worst_ndcg@10": wide.loc[qid, worst],
                         "n_relevant": int(sel.n_relevant.iloc[0]), "relevant_ids": sel.relevant.iloc[0],
                         "top10_best": get(best).top10, "top10_worst": get(worst).top10})
    return pd.DataFrame(rows).head(n)


def doc_type_summary(res, docstats):
    rows = []
    for ds, g in res[res.metric == "ndcg@10"].groupby("dataset"):
        agg = g.groupby(["strategy", "chunk_size"])["value"].mean()
        chunked = agg.drop(index=(C.BASELINE, 0), errors="ignore")
        (bs, bz), bv = chunked.idxmax(), chunked.max()
        base = agg.get((C.BASELINE, 0), np.nan)
        fixed_best = agg[[i for i in agg.index if i[0] == "fixed"]].max()
        d = docstats[docstats.dataset == ds].mean(numeric_only=True)
        rows.append({"dataset": DS_NAME[ds], "mean_doc_tokens": round(d.mean_doc_tokens), "median_doc_tokens": round(d.median_doc_tokens),
                     **{f"pct_docs_over_{s}_tokens": round(d[f"pct_docs_over_{s}"], 1) for s in C.CHUNK_SIZES},
                     "best_chunked_config": f"{bs} ({bz})", "best_chunked_ndcg@10": round(bv, 4),
                     "no_chunking_ndcg@10": round(base, 4), "best_fixed_ndcg@10": round(fixed_best, 4),
                     "best_minus_no_chunking": round(bv - base, 4), "best_minus_best_fixed": round(bv - fixed_best, 4)})
    return pd.DataFrame(rows)


def main():
    R = C.RESULTS
    res = pd.read_csv(R / "results.csv")
    cost = pd.read_csv(R / "index_cost.csv")
    pq = pd.read_csv(R / "per_query.csv.gz", keep_default_na=False)
    pq["seed"] = pq["seed"].astype(int)
    docstats = pd.read_csv(R / "doc_stats.csv")

    t1, t2, t3 = table1(res), table2(cost), table3_ci(pq)
    t1.to_csv(R / "T1_quality.csv", index=False)
    t2.to_csv(R / "T2_cost.csv", index=False)
    t3.to_csv(R / "T3_ndcg_ci.csv", index=False)
    summ = doc_type_summary(res, docstats)
    summ.to_csv(R / "doc_type_summary.csv", index=False)
    failure_cases(pq).to_csv(R / "failure_cases.csv", index=False)
    (R / "tables.md").write_text(
        "## T1: retrieval quality (mean ± std over seeds; NFCorpus is deterministic, single run)\n\n" + t1.to_markdown(index=False)
        + "\n\n## T2: chunks and indexing time\n\n" + t2.to_markdown(index=False)
        + "\n\n## T3: NDCG@10 bootstrap CI\n\n" + t3.to_markdown(index=False)
        + "\n\n## Document type summary\n\n" + summ.to_markdown(index=False) + "\n", encoding="utf-8")
    figure1(res, R / "F1_ndcg_by_strategy.png")
    figure2(res, R / "F2_recall_curves.png")
    figure3_hotpot(pq, R / "F3_hotpot_by_qtype.png")
    print(t1.to_string(index=False)); print(); print(t2.to_string(index=False)); print(); print(summ.T.to_string())


if __name__ == "__main__":
    main()
