"""Step 5: GraphRAG local and global search over the 50 questions, per seed.

Usage:  python scripts/05_graphrag_query.py [--methods local global] [--limit N]
Requires LM Studio + usage proxy running and a finished index (step 4).
Token usage per query is the delta of the proxy's counters (queries run sequentially).
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
import httpx
import pandas as pd
from common import load_questions, write_jsonl
from graphrag.api import global_search, local_search
from graphrag.config.load_config import load_config

PROXY = "http://localhost:1235"


def stats():
    return httpx.get(f"{PROXY}/_stats", timeout=5).json()


def _col(df, names):
    for n in names:
        if n in df.columns:
            return df[n].astype(str).tolist()
    return []


def subgraph_from_context(ctx):
    """Entities/relationships GraphRAG put into the local-search prompt (used for F2)."""
    if not isinstance(ctx, dict):
        return {"entities": [], "relationships": []}
    ent = ctx.get("entities")
    rel = ctx.get("relationships")
    ents = _col(ent, ["entity", "title"]) if isinstance(ent, pd.DataFrame) else []
    rels = []
    if isinstance(rel, pd.DataFrame) and {"source", "target"} <= set(rel.columns):
        rels = [{"source": s, "target": t,
                 "description": d} for s, t, d in zip(
                     rel["source"].astype(str), rel["target"].astype(str),
                     rel["description"].astype(str) if "description" in rel else [""] * len(rel))]
    return {"entities": ents, "relationships": rels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["local", "global"], choices=["local", "global"])
    ap.add_argument("--limit", type=int, default=None, help="only first N questions (smoke test)")
    a = ap.parse_args()
    try:
        stats()
    except Exception as e:
        sys.exit(f"Usage proxy not reachable ({e}). Start: python scripts/usage_proxy.py")

    out = C.GRAPHRAG_ROOT / "output"
    ent = pd.read_parquet(out / "entities.parquet")
    comm = pd.read_parquet(out / "communities.parquet")
    reports = pd.read_parquet(out / "community_reports.parquet")
    units = pd.read_parquet(out / "text_units.parquet")
    rels = pd.read_parquet(out / "relationships.parquet")
    questions = load_questions()[: a.limit]
    loop = asyncio.new_event_loop()

    for method in a.methods:
        rows = []
        for seed in C.SEEDS:
            cfg = load_config(root_dir=C.GRAPHRAG_ROOT)
            cfg.completion_models["default_completion_model"].call_args = {
                "temperature": C.TEMPERATURE, "max_tokens": 1024, "seed": seed}
            for q in questions:
                before, t0 = stats(), time.time()
                try:
                    if method == "local":
                        resp, ctx = loop.run_until_complete(local_search(
                            config=cfg, entities=ent, communities=comm, community_reports=reports,
                            text_units=units, relationships=rels, covariates=None,
                            community_level=C.COMM_LEVEL_LOCAL, response_type=C.RESPONSE_TYPE, query=q["query"]))
                    else:
                        resp, ctx = loop.run_until_complete(global_search(
                            config=cfg, entities=ent, communities=comm, community_reports=reports,
                            community_level=C.COMM_LEVEL_GLOBAL, dynamic_community_selection=False,
                            response_type=C.RESPONSE_TYPE, query=q["query"]))
                    err = None
                except Exception as e:                       # keep going; count as wrong, report it
                    resp, ctx, err = "", {}, f"{type(e).__name__}: {e}"
                after = stats()
                rows.append({
                    "id": q["id"], "type": q["type"], "method": f"graphrag_{method}", "seed": seed,
                    "pred": resp if isinstance(resp, str) else str(resp), "gold": q["answer"],
                    "latency_s": time.time() - t0, "error": err,
                    "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
                    "completion_tokens": after["completion_tokens"] - before["completion_tokens"],
                    "embedding_tokens": after["embedding_tokens"] - before["embedding_tokens"],
                    "llm_calls": after["llm_calls"] - before["llm_calls"],
                    "subgraph": subgraph_from_context(ctx) if method == "local" else None})
                print(f"{method} seed {seed} {q['id']}: {str(resp)[:60]!r} | gold {q['answer']!r}"
                      + (f"  ERROR {err}" if err else ""))
        write_jsonl(C.LOCAL_ANSWERS if method == "local" else C.GLOBAL_ANSWERS, rows)


if __name__ == "__main__":
    main()
