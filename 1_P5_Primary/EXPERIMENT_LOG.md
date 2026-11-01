# Experiment log (P5 Primary)

Record what you actually ran and when. Do not edit numbers after the fact.

| Date | What was run | Command / settings | Outcome |
|---|---|---|---|
| 2026-10-26 | Project scaffolded with Claude Code (AI tool use): subset builder, naive RAG, GraphRAG config/index/query scripts, token proxy, evaluation, Streamlit app | see `scripts/` | Subset built (50 questions: 17 inference / 17 comparison / 16 temporal; 200 docs). Proxy and evaluation code smoke-tested on mock/synthetic data only; no real experiment run yet. |
| | LM Studio models loaded | LLM = ?, embedder = ?, quant = ?, ctx = ? | |
| | `02_naive_rag.py` | | |
| | `04_graphrag_index.py` | | |
| | `05_graphrag_query.py` | | |
| | `06_evaluate.py` | | |

## AI tool use
- Claude Code (Anthropic) generated the initial code in this folder. Code must be reviewed and run by the student; all reported numbers must come from real runs.

## Deviations from default GraphRAG
(List any prompt or setting changes made while getting a 3B model to index successfully.)
