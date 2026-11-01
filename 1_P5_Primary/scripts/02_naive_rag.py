"""Step 2: naive RAG baseline (chunk -> embed -> FAISS top-k -> LLM).

Usage:  python scripts/02_naive_rag.py [--skip-index]
Needs LM Studio serving the LLM and the embedder.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import tiktoken
import faiss
import config as C
from common import chat, embed, load_questions, write_jsonl

ENC = tiktoken.get_encoding("cl100k_base")


def chunk_text(text, size, overlap):
    ids = ENC.encode(text, disallowed_special=())
    step = size - overlap
    return [ENC.decode(ids[i:i + size]) for i in range(0, max(len(ids) - overlap, 1), step)]


def build_index():
    C.NAIVE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    chunks, meta = [], []
    for p in sorted(C.CORPUS_DIR.glob("*.txt")):
        for j, ch in enumerate(chunk_text(p.read_text(encoding="utf-8"), C.CHUNK_TOKENS, C.CHUNK_OVERLAP)):
            chunks.append(ch)
            meta.append({"doc_id": p.stem, "chunk": j})
    vecs, emb_tokens = embed(chunks)
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(C.NAIVE_DIR / "faiss.index"))
    pickle.dump({"chunks": chunks, "meta": meta}, open(C.NAIVE_DIR / "chunks.pkl", "wb"))
    elapsed = time.time() - t0

    stats = json.loads(C.INDEX_STATS.read_text()) if C.INDEX_STATS.exists() else {}
    stats["naive"] = {"indexing_seconds": elapsed, "llm_prompt_tokens": 0, "llm_completion_tokens": 0,
                      "embedding_tokens": emb_tokens, "num_chunks": len(chunks),
                      "num_nodes": None, "num_edges": None}
    C.RESULTS_DIR.mkdir(exist_ok=True)
    C.INDEX_STATS.write_text(json.dumps(stats, indent=2))
    print(f"indexed {len(chunks)} chunks in {elapsed:.1f}s ({emb_tokens} embedding tokens)")


def run_queries():
    index = faiss.read_index(str(C.NAIVE_DIR / "faiss.index"))
    store = pickle.load(open(C.NAIVE_DIR / "chunks.pkl", "rb"))
    rows = []
    for seed in C.SEEDS:
        for q in load_questions():
            t0 = time.time()
            qv, q_emb_tokens = embed([q["query"]])
            faiss.normalize_L2(qv)
            _, I = index.search(qv, C.TOP_K)
            ctx = [store["chunks"][i] for i in I[0]]
            context = "\n\n".join(f"[{n + 1}] {c}" for n, c in enumerate(ctx))
            msgs = [{"role": "system", "content": C.ANSWER_INSTRUCTION},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q['query']}\nAnswer:"}]
            pred, u = chat(msgs, seed=seed)
            rows.append({"id": q["id"], "type": q["type"], "method": "naive", "seed": seed,
                         "pred": pred, "gold": q["answer"], "latency_s": time.time() - t0,
                         "prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"],
                         "embedding_tokens": q_emb_tokens,
                         "retrieved_docs": [store["meta"][i]["doc_id"] for i in I[0]]})
            print(f"seed {seed} {q['id']}: {pred[:60]!r} | gold {q['answer']!r}")
    write_jsonl(C.NAIVE_ANSWERS, rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-index", action="store_true")
    a = ap.parse_args()
    if not a.skip_index:
        build_index()
    run_queries()
