# KGIRS mini-project

Two separate experiments from the course micro-project brief. Each folder follows the submission layout
(code, README, results CSV, experiment log, report folder).

| Folder | Project | Role | Status |
|---|---|---|---|
| [28_P1_Variant](28_P1_Variant) | P1 Chunking benchmark: fixed / recursive / semantic chunking on NFCorpus and a HotpotQA subset | Variant | Experiment run on a Colab T4; results in `28_P1_Variant/results`. Report not written yet. |
| [1_P5_Primary](1_P5_Primary) | P5 Naive RAG vs Microsoft GraphRAG on MultiHop-RAG | Primary | **Incomplete.** Naive RAG baseline done (local LM Studio run and a Colab run); GraphRAG indexing/query results do not exist yet. |

## Layout
```
28_P1_Variant/
  p1/                code package (config, data, chunkers, metrics, run_experiment, report)
  colab/             self-contained Colab notebook + its builder
  results/           results.csv, T1/T2/T3 tables, F1/F2/F3 figures, failure cases, run_info.json
  README.md, EXPERIMENT_LOG.md, report/
1_P5_Primary/
  scripts/           pipeline steps 01..06, token proxy, embedding server
  app/               Streamlit viewer
  colab/             Colab notebook + builder
  config.py, README.md, EXPERIMENT_LOG.md, requirements.txt
  data/              50-question subset and 200-document corpus
  results/           local LM Studio naive-RAG run
  results_colab_run1_naive_only/   first Colab download (GraphRAG never finished)
```
See each folder's README for how to run it.
