# KGIRS mini-project

## P1 Chunking benchmark (Variant): [28_P1_Variant](28_P1_Variant)
Compares fixed-length, recursive and semantic chunking (128 and 256 tokens) against a no-chunking baseline
on NFCorpus and a HotpotQA subset, using BGE-small. Run on a Colab T4; results are in `28_P1_Variant/results`.

```
28_P1_Variant/
  p1/          code package (config, data, chunkers, metrics, run_experiment, report)
  colab/       self-contained Colab notebook + its builder
  results/     results.csv, T1/T2/T3 tables, F1/F2/F3 figures, failure cases, run_info.json
  report/      (report to be added)
  README.md, EXPERIMENT_LOG.md
```
See `28_P1_Variant/README.md` for how to run it.
