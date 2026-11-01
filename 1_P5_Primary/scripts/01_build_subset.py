"""Step 1: build the MultiHop-RAG subset (<=200 docs, 50 questions).

Questions are sampled with a fixed seed, stratified by question type. The
corpus subset = gold evidence articles of the chosen questions + random
distractor articles up to MAX_DOCS.
"""
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from datasets import load_dataset


def main():
    rng = random.Random(C.SUBSET_SEED)
    qs = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")
    corpus = load_dataset("yixuantt/MultiHopRAG", "corpus", split="train")

    docs = [dict(d, doc_id=f"doc{i:03d}") for i, d in enumerate(corpus)]
    by_title = {d["title"].strip(): d["doc_id"] for d in docs}
    by_url = {d["url"].strip(): d["doc_id"] for d in docs}

    def evidence_ids(q):
        ids = []
        for ev in q["evidence_list"]:
            did = by_title.get(ev["title"].strip()) or by_url.get(ev["url"].strip())
            if did is None:
                return None
            if did not in ids:
                ids.append(did)
        return ids

    types = ["inference_query", "comparison_query", "temporal_query"]
    if C.INCLUDE_NULL_QUERIES:
        types.append("null_query")
    base, extra = divmod(C.N_QUESTIONS, len(types))
    quota = {t: base + (1 if i < extra else 0) for i, t in enumerate(types)}

    pool = defaultdict(list)
    for q in qs:
        if q["question_type"] not in quota:
            continue
        ids = evidence_ids(q) if q["question_type"] != "null_query" else []
        if ids is None:
            continue
        pool[q["question_type"]].append((q, ids))
    for t in pool:
        rng.shuffle(pool[t])

    chosen, doc_set = [], set()
    for t in types:
        taken = 0
        for q, ids in pool[t]:
            if taken >= quota[t]:
                break
            if len(doc_set | set(ids)) > C.MAX_DOCS - 10:   # keep room for distractors
                continue
            doc_set |= set(ids)
            chosen.append((q, ids))
            taken += 1
        if taken < quota[t]:
            print(f"WARNING: only {taken}/{quota[t]} questions for {t}")

    distractors = [d["doc_id"] for d in docs if d["doc_id"] not in doc_set]
    rng.shuffle(distractors)
    for did in distractors:
        if len(doc_set) >= C.MAX_DOCS:
            break
        doc_set.add(did)

    if C.CORPUS_DIR.exists():
        shutil.rmtree(C.CORPUS_DIR)
    C.CORPUS_DIR.mkdir(parents=True)
    by_id = {d["doc_id"]: d for d in docs}
    for did in sorted(doc_set):
        d = by_id[did]
        text = f"{d['title']}\n{d['source']} | {d['published_at']} | {d['category']}\n\n{d['body']}"
        (C.CORPUS_DIR / f"{did}.txt").write_text(text, encoding="utf-8")

    out = []
    for i, (q, ids) in enumerate(chosen):
        out.append({
            "id": f"q{i:02d}",
            "query": q["query"],
            "answer": q["answer"],
            "type": q["question_type"].replace("_query", ""),
            "gold_doc_ids": ids,
            "evidence_facts": [e["fact"] for e in q["evidence_list"]],
        })
    C.DATA_DIR.mkdir(exist_ok=True)
    C.QUESTIONS_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"questions: {len(out)}  by type: {dict(Counter(o['type'] for o in out))}")
    print(f"documents: {len(doc_set)} (gold evidence: {len(doc_set) - sum(1 for d in doc_set if d in set(distractors))})")


if __name__ == "__main__":
    main()
