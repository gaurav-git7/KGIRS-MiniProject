"""Step 4: run GraphRAG indexing and record time, tokens and graph size (T2).

Requires: LM Studio server running AND `python scripts/usage_proxy.py` running.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
import httpx
import pandas as pd

PROXY = "http://localhost:1235"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="keep GraphRAG cache/output after an interrupted run (time and tokens then cover only the resumed part)")
    a = ap.parse_args()
    try:
        httpx.post(f"{PROXY}/_reset", timeout=5)
    except Exception as e:
        sys.exit(f"Usage proxy not reachable ({e}). Start: python scripts/usage_proxy.py")

    out = C.GRAPHRAG_ROOT / "output"
    if not a.resume:
        for d in (out, C.GRAPHRAG_ROOT / "cache", C.GRAPHRAG_ROOT / "logs"):
            shutil.rmtree(d, ignore_errors=True)   # cold index: no cache hits skew time/tokens

    t0 = time.time()
    subprocess.run([sys.executable, "-m", "graphrag", "index", "--root", str(C.GRAPHRAG_ROOT)], check=True)
    elapsed = time.time() - t0
    u = httpx.get(f"{PROXY}/_stats", timeout=5).json()

    ent = pd.read_parquet(out / "entities.parquet")
    rel = pd.read_parquet(out / "relationships.parquet")
    comm = pd.read_parquet(out / "communities.parquet")
    rep = pd.read_parquet(out / "community_reports.parquet")
    units = pd.read_parquet(out / "text_units.parquet")

    stats = json.loads(C.INDEX_STATS.read_text()) if C.INDEX_STATS.exists() else {}
    stats["graphrag"] = {
        "indexing_seconds": elapsed,
        "llm_prompt_tokens": u["prompt_tokens"], "llm_completion_tokens": u["completion_tokens"],
        "embedding_tokens": u["embedding_tokens"], "llm_calls": u["llm_calls"],
        "estimated_calls": u["estimated_calls"],
        "num_nodes": int(len(ent)), "num_edges": int(len(rel)),
        "num_communities": int(len(comm)), "num_community_reports": int(len(rep)),
        "num_chunks": int(len(units)),
        "resumed_run": int(a.resume),   # 1 = time/tokens undercount the full cold index
    }
    C.RESULTS_DIR.mkdir(exist_ok=True)
    C.INDEX_STATS.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats["graphrag"], indent=2))


if __name__ == "__main__":
    main()
