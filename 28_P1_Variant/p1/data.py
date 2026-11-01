"""Dataset loaders. Each returns a dict:
    corpus : {doc_id: {"title": str, "text": str}}
    queries: {qid: str}
    qrels  : {qid: {doc_id: graded relevance (>0)}}
    qtype  : {qid: str}   (question category, may be empty)
"""
import random

from datasets import load_dataset

from . import config as C


def load_nfcorpus(max_queries=None):
    corpus_ds = load_dataset("BeIR/nfcorpus", "corpus")["corpus"]
    query_ds = load_dataset("BeIR/nfcorpus", "queries")["queries"]
    qrel_ds = load_dataset("BeIR/nfcorpus-qrels")["test"]

    qrels = {}
    for r in qrel_ds:
        if r["score"] > 0:
            qrels.setdefault(r["query-id"], {})[r["corpus-id"]] = int(r["score"])
    queries = {q["_id"]: q["text"] for q in query_ds if q["_id"] in qrels}
    qids = sorted(queries)
    if max_queries:
        qids = qids[:max_queries]
    corpus = {d["_id"]: {"title": d["title"] or "", "text": d["text"] or ""} for d in corpus_ds}
    qrels = {q: {d: s for d, s in qrels[q].items() if d in corpus} for q in qids}
    qids = [q for q in qids if qrels[q]]
    if C.SMOKE:        # testing only: keep relevant docs plus 300 distractors so CPU runs finish quickly
        keep = {d for q in qids for d in qrels[q]}
        extra = [d for d in sorted(corpus) if d not in keep][:300]
        corpus = {d: corpus[d] for d in sorted(keep) + extra}
    return {"corpus": corpus, "queries": {q: queries[q] for q in qids},
            "qrels": {q: qrels[q] for q in qids}, "qtype": {q: "" for q in qids}}


def load_hotpotqa(seed, n_questions):
    """Random sample of HotpotQA (distractor setting, validation split).

    Corpus = every context paragraph of the sampled questions (gold + distractors), de-duplicated
    by Wikipedia title. Relevant documents = the paragraphs named in `supporting_facts`.
    """
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    corpus, queries, qrels, qtype = {}, {}, {}, {}
    for i in idx[:n_questions]:
        ex = ds[i]
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            corpus.setdefault(title, {"title": title, "text": "".join(sents).strip()})
        qid = ex["id"]
        queries[qid] = ex["question"]
        qrels[qid] = {t: 1 for t in set(ex["supporting_facts"]["title"]) if t in corpus}
        qtype[qid] = ex["type"]                       # bridge | comparison
    return {"corpus": corpus, "queries": queries, "qrels": qrels, "qtype": qtype}


def load(dataset, seed=0):
    if dataset == "nfcorpus":
        return load_nfcorpus(C.NFCORPUS_MAX_QUERIES)
    if dataset == "hotpotqa":
        return load_hotpotqa(seed, C.HOTPOT_N_QUESTIONS)
    raise ValueError(dataset)


def doc_text(d):
    """Text that gets chunked: title and body, as in BEIR's title + text convention."""
    return f"{d['title']}\n{d['text']}".strip() if d["title"] else d["text"]
