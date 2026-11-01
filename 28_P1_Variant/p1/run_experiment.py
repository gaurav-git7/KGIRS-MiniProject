"""Run the whole chunking benchmark:  python -m p1.run_experiment

For every (dataset, seed): build chunks with each strategy x chunk size (plus the no-chunking
baseline), embed them with BGE-small, retrieve, collapse chunk hits to documents (best chunk
score per document) and score Recall@1..20, MRR@10 and NDCG@10.

Writes to results/: results.csv, per_query.csv.gz, index_cost.csv, doc_stats.csv, run_info.json
"""
import json
import platform
import random
import time

import numpy as np
import pandas as pd
import torch

from . import config as C
from . import data as D
from .chunkers import SemanticState, Tok, build_chunks
from .metrics import mrr_at_k, ndcg_at_k, recall_at_k


class Embedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(C.EMBED_MODEL, device=self.device)
        self.model.max_seq_length = C.MAX_SEQ
        self.encode_texts(["warm up the model"] * 8)          # keep CUDA start-up out of the timings

    def encode_texts(self, texts, batch=128):
        e = self.model.encode(texts, batch_size=batch, normalize_embeddings=True,
                              convert_to_tensor=True, show_progress_bar=False)
        if self.device == "cuda":
            torch.cuda.synchronize()
        return e

    def encode_queries(self, queries):
        return self.encode_texts([C.QUERY_PREFIX + q for q in queries])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rank_documents(qemb, cemb, owner_idx, n_docs, depth):
    """Best-chunk-per-document ranking. Returns (Q, depth) tensor of document indices."""
    scores = qemb @ cemb.T                                           # (Q, n_chunks)
    doc_scores = torch.full((scores.shape[0], n_docs), -1e9, device=scores.device)
    doc_scores.scatter_reduce_(1, owner_idx.unsqueeze(0).expand_as(scores), scores, reduce="amax")
    return doc_scores.topk(min(depth, n_docs), dim=1).indices.cpu().numpy()


def main():
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    tok, emb = Tok(), Embedder()
    results, per_query, costs, docstats = [], [], [], []

    for ds_name in C.DATASETS:
        seeds = C.SEEDS if ds_name == "hotpotqa" else [0]
        for seed in seeds:
            seed_all(seed)
            ds = D.load(ds_name, seed)
            doc_ids = list(ds["corpus"])
            docs = {d: D.doc_text(ds["corpus"][d]) for d in doc_ids}
            titles = {d: ds["corpus"][d]["title"] for d in doc_ids}
            owner_of = {d: i for i, d in enumerate(doc_ids)}
            qids = list(ds["queries"])
            qemb = emb.encode_queries([ds["queries"][q] for q in qids])
            print(f"\n== {ds_name} seed {seed}: {len(doc_ids)} docs, {len(qids)} queries ==", flush=True)

            dtok = np.array(tok.counts([docs[d] for d in doc_ids]))
            docstats.append({"dataset": ds_name, "seed": seed, "n_docs": len(doc_ids), "n_queries": len(qids),
                             "mean_doc_tokens": dtok.mean(), "median_doc_tokens": float(np.median(dtok)),
                             **{f"pct_docs_over_{s}": float((dtok > s).mean() * 100) for s in C.CHUNK_SIZES + [512]}})

            sem_state = SemanticState(docs, tok, emb) if "semantic" in C.STRATEGIES else None
            grid = [(C.BASELINE, 0)] + [(s, z) for s in C.STRATEGIES for z in C.CHUNK_SIZES]
            for strategy, size in grid:
                texts, owners, chunk_s = build_chunks(docs, titles, strategy, size, tok, sem_state)
                t0 = time.time()
                cemb = emb.encode_texts(texts)
                embed_s = time.time() - t0
                owner_idx = torch.tensor([owner_of[o] for o in owners], device=cemb.device)
                ranked = rank_documents(qemb, cemb, owner_idx, len(doc_ids), C.DEPTH)
                ntok = np.array(tok.counts(texts))
                costs.append({"dataset": ds_name, "seed": seed, "strategy": strategy, "chunk_size": size,
                              "n_docs": len(doc_ids), "n_chunks": len(texts), "mean_chunk_tokens": ntok.mean(),
                              "chunk_seconds": chunk_s, "embed_seconds": embed_s,
                              "index_seconds": chunk_s + embed_s})

                curves, mrrs, ndcgs = [], [], []
                for qi, q in enumerate(qids):
                    rel = ds["qrels"][q]
                    r = [doc_ids[j] for j in ranked[qi]]
                    rec = [recall_at_k(r, rel, k) for k in C.KS]
                    m, n = mrr_at_k(r, rel, 10), ndcg_at_k(r, rel, 10)
                    curves.append(rec); mrrs.append(m); ndcgs.append(n)
                    per_query.append({"dataset": ds_name, "seed": seed, "strategy": strategy, "chunk_size": size,
                                      "qid": q, "qtype": ds["qtype"][q], "n_relevant": len(rel),
                                      "recall@5": rec[4], "recall@10": rec[9], "mrr@10": m, "ndcg@10": n,
                                      "top10": "|".join(r[:10]), "relevant": "|".join(rel)})
                curve = np.mean(curves, axis=0)
                base = {"dataset": ds_name, "seed": seed, "strategy": strategy, "chunk_size": size, "n_queries": len(qids)}
                for k, v in zip(C.KS, curve):
                    results.append({**base, "metric": f"recall@{k}", "value": float(v)})
                results.append({**base, "metric": "mrr@10", "value": float(np.mean(mrrs))})
                results.append({**base, "metric": "ndcg@10", "value": float(np.mean(ndcgs))})
                for mname in ("n_chunks", "index_seconds"):
                    results.append({**base, "metric": mname, "value": float(costs[-1][mname])})
                print(f"{strategy:9s} size {size:3d}  chunks {len(texts):6d}  R@5 {curve[4]:.3f}  R@10 {curve[9]:.3f}  "
                      f"MRR {np.mean(mrrs):.3f}  NDCG@10 {np.mean(ndcgs):.3f}  index {chunk_s + embed_s:.1f}s", flush=True)
                del cemb
                if emb.device == "cuda":
                    torch.cuda.empty_cache()

    pd.DataFrame(results).to_csv(C.RESULTS / "results.csv", index=False)
    pd.DataFrame(per_query).to_csv(C.RESULTS / "per_query.csv.gz", index=False)
    pd.DataFrame(costs).to_csv(C.RESULTS / "index_cost.csv", index=False)
    pd.DataFrame(docstats).to_csv(C.RESULTS / "doc_stats.csv", index=False)
    import sentence_transformers, transformers, datasets
    info = {"embed_model": C.EMBED_MODEL, "device": emb.device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python": platform.python_version(), "torch": torch.__version__,
            "sentence_transformers": sentence_transformers.__version__, "transformers": transformers.__version__,
            "datasets": datasets.__version__, "smoke": C.SMOKE,
            "config": {k: getattr(C, k) for k in ("DATASETS", "STRATEGIES", "CHUNK_SIZES", "CHUNK_OVERLAP",
                                                  "PREPEND_TITLE", "SEMANTIC_PERCENTILE", "SEMANTIC_MIN_FRAC",
                                                  "SEEDS", "HOTPOT_N_QUESTIONS", "NFCORPUS_MAX_QUERIES", "DEPTH")}}
    (C.RESULTS / "run_info.json").write_text(json.dumps(info, indent=2))
    print("\nwrote results to", C.RESULTS)


if __name__ == "__main__":
    main()
