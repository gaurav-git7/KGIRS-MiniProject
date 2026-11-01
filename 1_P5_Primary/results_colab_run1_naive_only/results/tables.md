## T1

| Method                          | Accuracy (%)   | EM (%)     | F1 (%)     | Inference (%)   | Comparison (%)   | Temporal (%)   |
|:--------------------------------|:---------------|:-----------|:-----------|:----------------|:-----------------|:---------------|
| Naive RAG (baseline) [baseline] | 44.0 ± 0.0     | 43.3 ± 1.2 | 43.8 ± 0.4 | 90.2 ± 3.4      | 21.6 ± 3.4       | 18.8 ± 0.0     |

## T2

| Method               | Indexing time (s)   | Index LLM tokens   | Index embedding tokens   | Nodes   | Edges   | Chunks   | Query LLM tokens / question   | Latency / question (s)   |
|:---------------------|:--------------------|:-------------------|:-------------------------|:--------|:--------|:---------|:------------------------------|:-------------------------|
| Naive RAG (baseline) | 20                  | 0                  | 592,228                  | –       | –       | 2058     | 1,641                         | 0.5                      |
| GraphRAG local       | n/a                 | n/a                | n/a                      | n/a     | n/a     | n/a      | n/a                           | n/a                      |
| GraphRAG global      | n/a                 | n/a                | n/a                      | n/a     | n/a     | n/a      | n/a                           | n/a                      |
