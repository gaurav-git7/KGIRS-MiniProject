## T1

| Method                          | Accuracy (%)   | EM (%)     | F1 (%)     | Inference (%)   | Comparison (%)   | Temporal (%)   |
|:--------------------------------|:---------------|:-----------|:-----------|:----------------|:-----------------|:---------------|
| Naive RAG (baseline) [baseline] | 38.0 ± 0.0     | 36.7 ± 1.2 | 36.7 ± 1.1 | 86.3 ± 3.4      | 13.7 ± 3.4       | 12.5 ± 0.0     |

## T2

| Method               | Indexing time (s)   | Index LLM tokens   | Index embedding tokens   | Nodes   | Edges   | Chunks   | Query LLM tokens / question   | Latency / question (s)   |
|:---------------------|:--------------------|:-------------------|:-------------------------|:--------|:--------|:---------|:------------------------------|:-------------------------|
| Naive RAG (baseline) | 20                  | 0                  | 592,228                  | –       | –       | 2058     | 1,643                         | 0.5                      |
| GraphRAG local       | n/a                 | n/a                | n/a                      | n/a     | n/a     | n/a      | n/a                           | n/a                      |
| GraphRAG global      | n/a                 | n/a                | n/a                      | n/a     | n/a     | n/a      | n/a                           | n/a                      |
