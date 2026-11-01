## T1: retrieval quality (mean ± std over seeds; NFCorpus is deterministic, single run)

| Dataset         | Strategy                          | Recall@5      | Recall@10     | MRR@10        | NDCG@10       |
|:----------------|:----------------------------------|:--------------|:--------------|:--------------|:--------------|
| HotpotQA subset | No chunking (baseline) [baseline] | 0.871 ± 0.004 | 0.922 ± 0.004 | 0.948 ± 0.011 | 0.873 ± 0.006 |
| HotpotQA subset | Fixed-length (128)                | 0.840 ± 0.005 | 0.907 ± 0.005 | 0.921 ± 0.011 | 0.843 ± 0.005 |
| HotpotQA subset | Fixed-length (256)                | 0.872 ± 0.006 | 0.921 ± 0.003 | 0.947 ± 0.013 | 0.873 ± 0.007 |
| HotpotQA subset | Recursive (128)                   | 0.870 ± 0.004 | 0.919 ± 0.004 | 0.947 ± 0.012 | 0.871 ± 0.006 |
| HotpotQA subset | Recursive (256)                   | 0.871 ± 0.006 | 0.922 ± 0.004 | 0.948 ± 0.011 | 0.873 ± 0.006 |
| HotpotQA subset | Semantic (128)                    | 0.867 ± 0.004 | 0.920 ± 0.003 | 0.941 ± 0.011 | 0.866 ± 0.004 |
| HotpotQA subset | Semantic (256)                    | 0.869 ± 0.003 | 0.920 ± 0.006 | 0.947 ± 0.010 | 0.871 ± 0.005 |
| NFCorpus        | No chunking (baseline) [baseline] | 0.127         | 0.162         | 0.530         | 0.344         |
| NFCorpus        | Fixed-length (128)                | 0.124         | 0.165         | 0.538         | 0.341         |
| NFCorpus        | Fixed-length (256)                | 0.131         | 0.170         | 0.530         | 0.347         |
| NFCorpus        | Recursive (128)                   | 0.132         | 0.172         | 0.543         | 0.350         |
| NFCorpus        | Recursive (256)                   | 0.132         | 0.167         | 0.540         | 0.352         |
| NFCorpus        | Semantic (128)                    | 0.136         | 0.170         | 0.551         | 0.353         |
| NFCorpus        | Semantic (256)                    | 0.131         | 0.166         | 0.548         | 0.350         |

## T2: chunks and indexing time

| Dataset         | Strategy                          | Chunks   |   Mean tokens/chunk |   Chunking time (s) |   Embedding time (s) |   Indexing time (s) |
|:----------------|:----------------------------------|:---------|--------------------:|--------------------:|---------------------:|--------------------:|
| HotpotQA subset | No chunking (baseline) [baseline] | 4,929    |                 123 |                0    |                16.37 |               16.37 |
| HotpotQA subset | Fixed-length (128)                | 7,074    |                  86 |                2.11 |                13.5  |               15.6  |
| HotpotQA subset | Fixed-length (256)                | 5,150    |                 118 |                2.04 |                14.92 |               16.96 |
| HotpotQA subset | Recursive (128)                   | 7,213    |                  84 |                9.5  |                13.74 |               23.24 |
| HotpotQA subset | Recursive (256)                   | 5,153    |                 118 |                4.88 |                16.2  |               21.08 |
| HotpotQA subset | Semantic (128)                    | 8,877    |                  68 |               20.14 |                15.49 |               35.63 |
| HotpotQA subset | Semantic (256)                    | 6,563    |                  92 |               20.12 |                14.83 |               34.95 |
| NFCorpus        | No chunking (baseline) [baseline] | 3,633    |                 355 |                0    |                31.81 |               31.81 |
| NFCorpus        | Fixed-length (128)                | 11,901   |                 109 |                3.63 |                27.37 |               31    |
| NFCorpus        | Fixed-length (256)                | 6,843    |                 189 |                3.6  |                28.61 |               32.21 |
| NFCorpus        | Recursive (128)                   | 13,340   |                  97 |               23.9  |                31.22 |               55.12 |
| NFCorpus        | Recursive (256)                   | 7,040    |                 183 |               21.65 |                32.31 |               53.96 |
| NFCorpus        | Semantic (128)                    | 15,689   |                  82 |               43.28 |                32.66 |               75.94 |
| NFCorpus        | Semantic (256)                    | 9,308    |                 139 |               43.21 |                35.53 |               78.73 |

## T3: NDCG@10 bootstrap CI

| Dataset         | Strategy               |   NDCG@10 | 95% bootstrap CI   |   Queries (pooled over seeds) |
|:----------------|:-----------------------|----------:|:-------------------|------------------------------:|
| HotpotQA subset | No chunking (baseline) |     0.873 | [0.864, 0.882]     |                          1500 |
| HotpotQA subset | Fixed-length (128)     |     0.843 | [0.833, 0.854]     |                          1500 |
| HotpotQA subset | Fixed-length (256)     |     0.873 | [0.863, 0.882]     |                          1500 |
| HotpotQA subset | Recursive (128)        |     0.871 | [0.861, 0.880]     |                          1500 |
| HotpotQA subset | Recursive (256)        |     0.873 | [0.864, 0.882]     |                          1500 |
| HotpotQA subset | Semantic (128)         |     0.866 | [0.857, 0.876]     |                          1500 |
| HotpotQA subset | Semantic (256)         |     0.871 | [0.862, 0.880]     |                          1500 |
| NFCorpus        | No chunking (baseline) |     0.344 | [0.307, 0.375]     |                           323 |
| NFCorpus        | Fixed-length (128)     |     0.341 | [0.304, 0.373]     |                           323 |
| NFCorpus        | Fixed-length (256)     |     0.347 | [0.311, 0.380]     |                           323 |
| NFCorpus        | Recursive (128)        |     0.35  | [0.312, 0.383]     |                           323 |
| NFCorpus        | Recursive (256)        |     0.352 | [0.314, 0.384]     |                           323 |
| NFCorpus        | Semantic (128)         |     0.353 | [0.315, 0.385]     |                           323 |
| NFCorpus        | Semantic (256)         |     0.35  | [0.311, 0.383]     |                           323 |

## Document type summary

| dataset         |   mean_doc_tokens |   median_doc_tokens |   pct_docs_over_128_tokens |   pct_docs_over_256_tokens | best_chunked_config   |   best_chunked_ndcg@10 |   no_chunking_ndcg@10 |   best_fixed_ndcg@10 |   best_minus_no_chunking |   best_minus_best_fixed |
|:----------------|------------------:|--------------------:|---------------------------:|---------------------------:|:----------------------|-----------------------:|----------------------:|---------------------:|-------------------------:|------------------------:|
| HotpotQA subset |               123 |                 111 |                       38.2 |                        4.2 | recursive (256)       |                 0.8731 |                0.8733 |               0.8725 |                  -0.0001 |                  0.0006 |
| NFCorpus        |               355 |                 355 |                       97.2 |                       78.3 | semantic (128)        |                 0.3529 |                0.3438 |               0.3471 |                   0.0091 |                  0.0057 |
