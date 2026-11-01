"""Central configuration for the P5 experiment (naive RAG vs GraphRAG local/global).

Everything that must be identical across the three systems lives here so the
comparison is fair: same LLM, same embedder, same seeds, same question subset.
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
GRAPHRAG_ROOT = ROOT / "graphrag_workspace"   # created by 03_graphrag_setup.py
NAIVE_DIR = RESULTS_DIR / "naive_index"

# ---- LM Studio (OpenAI-compatible server) -------------------------------
# Every endpoint/model setting can be overridden by an environment variable (used by the Colab notebook).
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL", LLM_BASE_URL)
LLM_API_KEY = "lm-studio"                      # LM Studio ignores the key
# Use the exact identifiers shown in LM Studio's "Developer" tab.
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5-3b-instruct")              # <= 3B parameters (project rule)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-bge-small-en-v1.5")   # small open embedder
EMBED_DIM = 384
LLM_CONTEXT = 8192                              # set the same value in LM Studio

# ---- Experiment ---------------------------------------------------------
GRAPHRAG_CONCURRENCY = int(os.environ.get("GRAPHRAG_CONCURRENCY", "4"))
SEEDS = [0, 1, 2]            # query-time seeds -> mean +/- std
SUBSET_SEED = 42             # question/doc sampling (fixed, not varied)
N_QUESTIONS = 50
MAX_DOCS = 200
INCLUDE_NULL_QUERIES = False  # null queries have no gold evidence; see README
TEMPERATURE = 0.3            # >0 so that seeds can differ; 0 gives identical runs
MAX_TOKENS = 256

# ---- Naive RAG ----------------------------------------------------------
CHUNK_TOKENS = 300           # GraphRAG is configured with the same chunk size
CHUNK_OVERLAP = 50
TOP_K = 5

# ---- GraphRAG query -----------------------------------------------------
COMM_LEVEL_LOCAL = 2
COMM_LEVEL_GLOBAL = 1        # coarser level = fewer community reports = cheaper map step
RESPONSE_TYPE = "a short answer of a few words (Yes/No for yes-no questions)"

# ---- Files --------------------------------------------------------------
QUESTIONS_JSON = DATA_DIR / "questions.json"
CORPUS_DIR = DATA_DIR / "corpus_subset"
NAIVE_ANSWERS = RESULTS_DIR / "answers_naive.jsonl"
LOCAL_ANSWERS = RESULTS_DIR / "answers_graphrag_local.jsonl"
GLOBAL_ANSWERS = RESULTS_DIR / "answers_graphrag_global.jsonl"
INDEX_STATS = RESULTS_DIR / "index_stats.json"
RESULTS_CSV = RESULTS_DIR / "results.csv"

ANSWER_INSTRUCTION = (
    "Answer the question using only the provided context. "
    "Reply with a short answer (a few words, or Yes/No for yes-no questions). "
    "If the context does not contain enough information, reply exactly: "
    "Insufficient information."
)
