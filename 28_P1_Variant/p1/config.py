"""Configuration for P1 (Variant): chunking benchmark on NFCorpus and a HotpotQA subset.

Environment overrides: P1_ROOT (output root), P1_SMOKE=1 (tiny run for testing only).
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("P1_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = ROOT / "results"

SMOKE = os.environ.get("P1_SMOKE") == "1"

# ---- models (open, small) -----------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # BGE retrieval instruction
MAX_SEQ = 512

# ---- experiment grid -----------------------------------------------------
DATASETS = ["nfcorpus", "hotpotqa"]
STRATEGIES = ["fixed", "recursive", "semantic"]
CHUNK_SIZES = [128, 256]            # max tokens per chunk (BGE tokenizer tokens)
CHUNK_OVERLAP = 0                   # tokens; kept 0 so strategies differ only in where they cut
PREPEND_TITLE = False               # True = prefix every chunk with the document title
BASELINE = "whole_doc"              # no chunking: whole document, truncated to MAX_SEQ by the model

# semantic chunking: cut where adjacent-sentence distance exceeds this corpus-wide percentile,
# subject to the max chunk size and a minimum chunk size (fraction of max)
SEMANTIC_PERCENTILE = 80
SEMANTIC_MIN_FRAC = 0.25

# ---- data ----------------------------------------------------------------
SEEDS = [0, 1, 2]                   # HotpotQA: three different question samples. NFCorpus is deterministic (one run).
HOTPOT_N_QUESTIONS = 500            # questions per seed; corpus = their context paragraphs
NFCORPUS_MAX_QUERIES = None         # None = all test queries with judgments (323)

# ---- evaluation ----------------------------------------------------------
KS = list(range(1, 21))             # recall curve K = 1..20
DEPTH = 100                         # ranked documents kept per query
BOOTSTRAP = 1000

if SMOKE:                           # never use for reported numbers
    SEEDS = [0]
    HOTPOT_N_QUESTIONS = 60
    NFCORPUS_MAX_QUERIES = 40
    BOOTSTRAP = 50
