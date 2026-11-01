"""Ranking metrics at document level (chunk hits are collapsed to their parent document)."""
import math

import numpy as np


def recall_at_k(ranked, rel, k):
    return len(set(ranked[:k]) & set(rel)) / len(rel)


def mrr_at_k(ranked, rel, k=10):
    for i, d in enumerate(ranked[:k], 1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked, rel, k=10):
    """Linear gain (gain = graded relevance), log2 discount, as in BEIR / trec_eval ndcg_cut."""
    dcg = sum(rel.get(d, 0) / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg / idcg if idcg > 0 else 0.0


def bootstrap_ci(values, n=1000, seed=0):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


if __name__ == "__main__":      # self-test against hand-computed values
    rel = {"a": 2, "b": 1, "c": 1}
    r = ["x", "a", "y", "b"]
    assert abs(recall_at_k(r, rel, 4) - 2 / 3) < 1e-9
    assert mrr_at_k(r, rel) == 0.5
    dcg = 2 / math.log2(3) + 1 / math.log2(5)
    idcg = 2 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert abs(ndcg_at_k(r, rel) - dcg / idcg) < 1e-9
    assert ndcg_at_k(["a", "b", "c"], rel) == 1.0
    print("metrics self-test OK")
