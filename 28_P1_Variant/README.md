# P1 Chunking benchmark (Variant): NFCorpus and a HotpotQA subset

Compares fixed-length, recursive and semantic chunking at 128 and 256 tokens, plus a no-chunking
baseline, using BAAI/bge-small-en-v1.5. Chunk hits are collapsed to documents (best chunk per document).

## Run
Easiest: upload `colab/P1_Variant_Colab.ipynb` to Colab, choose a T4 GPU, run all cells. It is self-contained.

Locally (GPU recommended):
```bash
pip install sentence-transformers datasets tabulate matplotlib pandas
python -m p1.metrics          # metric self-test
python -m p1.run_experiment   # results.csv, per_query.csv.gz, index_cost.csv, doc_stats.csv
python -m p1.report           # T1, T2, T3, F1, F2, F3, failure_cases.csv, doc_type_summary.csv
```
`P1_SMOKE=1` runs a tiny version for testing only; never report its numbers.

## Design decisions (state in the report)
- **Data:** NFCorpus (BEIR test queries with judgments, full corpus). HotpotQA distractor validation split: 500 random questions per seed; corpus = their context paragraphs; relevant = supporting-fact paragraphs.
- **Seeds:** HotpotQA uses 3 seeds = 3 different question samples (mean ± std). NFCorpus is deterministic, so it is run once; T3 gives bootstrap 95% CIs instead.
- **Sizes** are in BGE-tokenizer tokens; overlap 0; chunks are slices of the original text; titles are part of the document text (only the first chunk carries the title unless `PREPEND_TITLE`).
- **Semantic:** sentence embeddings with the same BGE model; cut where adjacent-sentence distance exceeds the corpus-wide 80th percentile, with the chunk size as a hard cap and a minimum of 25% of it. Sentence-embedding time is included in its indexing time.
- **Metrics:** Recall@K = share of relevant documents in the top K; MRR@10; NDCG@10 with linear graded gain.
- **Baseline:** whole document embedded without chunking (model truncates at 512 tokens).
