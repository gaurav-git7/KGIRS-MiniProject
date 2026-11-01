# P5 (Primary): naive RAG vs Microsoft GraphRAG on MultiHop-RAG

Compares naive RAG, GraphRAG local search and GraphRAG global search on 50 MultiHop-RAG
questions over at most 200 news articles, using one small open LLM and one small embedder
served by LM Studio. Reports accuracy (T1), indexing time / tokens / graph size (T2),
accuracy by question type (F1) and an example retrieved subgraph (F2).

## Versions (tested)
Python 3.12.9, graphrag 3.3.0, sentence-transformers 6.2.0 (unused at runtime; embeddings come
from LM Studio), faiss-cpu 1.15.1, openai 2.54.0, streamlit 1.66.1. Full list: `requirements.txt`.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

In **LM Studio** (Developer tab):
1. Load an LLM of at most 3B parameters (e.g. Qwen2.5-3B-Instruct, Q4_K_M or Q8_0), context length 8192.
2. Load an embedding model (e.g. `bge-small-en-v1.5` GGUF).
3. Turn off just-in-time model swapping so both stay resident (keeps timings honest).
4. Start the server on `localhost:1234`.
5. Put the **exact model identifiers** into `config.py` (`LLM_MODEL`, `EMBED_MODEL`).

## Run (in order)
```bash
python scripts/01_build_subset.py        # 50 questions, 200 docs -> data/
python scripts/usage_proxy.py            # keep running in a second terminal (token counter, :1235)
python scripts/02_naive_rag.py           # baseline index + answers (3 seeds)
python scripts/03_graphrag_setup.py      # writes graphrag_workspace/settings.yaml + input
python scripts/04_graphrag_index.py      # cold index; records time, tokens, nodes, edges
python scripts/05_graphrag_query.py --limit 3   # smoke test, then run without --limit
python scripts/06_evaluate.py            # results.csv, T1, T2, F1, F2, failure cases
streamlit run app/streamlit_app.py       # interactive viewer
```

Outputs are in `results/`: `results.csv` (all metrics and seeds), `T1_accuracy.csv`,
`T2_cost_and_graph.csv`, `tables.md`, `F1_accuracy_by_type.png`, `F2_example_subgraph.png`,
`failure_cases.csv`, plus raw `answers_*.jsonl`.

## Design decisions (state these in the report)
- **Same models everywhere.** One LLM and one embedder for all three systems; chunk size 300 tokens, overlap 50, in both naive RAG and GraphRAG.
- **Question subset.** 50 questions stratified over inference / comparison / temporal (17/17/16), seed 42. Null queries are excluded (`INCLUDE_NULL_QUERIES`) because they have no gold evidence. Corpus = gold evidence articles + random distractors up to 200.
- **Accuracy** = normalised gold answer appears as whole words in the prediction (yes/no must be the first word). EM and token-F1 are also reported. GraphRAG answers are verbose, so containment is the primary metric; this is lenient and should be stated as a limitation.
- **Seeds.** Indexing runs once (it is expensive). The three seeds vary only the query-time sampling (temperature 0.3, seed 0/1/2); mean ± std (sample std) reflects query-time variance only, not indexing variance.
- **Tokens.** Naive RAG uses LM Studio's reported `usage`. GraphRAG calls go through `usage_proxy.py`, which sums `usage` (falls back to a tiktoken estimate and counts those calls in `estimated_calls`).
- **GraphRAG settings changed from default:** `max_gleanings=0`, chunk 300/50, `community_reports.max_length=1200`, `concurrent_requests=4`, local search community level 2, global level 1, static community selection. Default prompts are unmodified unless noted in `EXPERIMENT_LOG.md`.
- **Failed GraphRAG queries** (e.g. a 3B model not following the map-step JSON format in global search) are kept as wrong answers with the error stored in `answers_*.jsonl`; they are not dropped.
