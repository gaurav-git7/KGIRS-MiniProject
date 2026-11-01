"""Shared helpers: LM Studio client, JSONL I/O, answer matching."""
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from openai import OpenAI

_client = None
_embed_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=C.LLM_BASE_URL, api_key=C.LLM_API_KEY, timeout=600)
    return _client


def embed_client() -> OpenAI:
    global _embed_client
    if _embed_client is None:
        _embed_client = OpenAI(base_url=C.EMBED_BASE_URL, api_key=C.LLM_API_KEY, timeout=600)
    return _embed_client


def chat(messages, seed=0, temperature=None, max_tokens=None):
    """One chat completion against LM Studio. Returns (text, usage dict)."""
    r = client().chat.completions.create(
        model=C.LLM_MODEL, messages=messages, seed=seed,
        temperature=C.TEMPERATURE if temperature is None else temperature,
        max_tokens=max_tokens or C.MAX_TOKENS)
    u = r.usage
    return (r.choices[0].message.content or "").strip(), {
        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or 0}


def embed(texts, batch=32):
    import numpy as np
    """Returns (matrix, embedding_tokens)."""
    out, tokens = [], 0
    for i in range(0, len(texts), batch):
        r = embed_client().embeddings.create(model=C.EMBED_MODEL, input=texts[i:i + batch])
        out.extend(d.embedding for d in r.data)
        reported = getattr(r.usage, "total_tokens", 0) or getattr(r.usage, "prompt_tokens", 0) or 0
        if not reported:      # LM Studio may omit embedding usage: estimate with tiktoken
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            reported = sum(len(enc.encode(t, disallowed_special=())) for t in texts[i:i + batch])
        tokens += reported
    return np.asarray(out, dtype="float32"), tokens


def load_questions():
    return json.loads(C.QUESTIONS_JSON.read_text(encoding="utf-8"))


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


# ---- answer matching ------------------------------------------------------
def normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation) - {"-"})
    return re.sub(r"\s+", " ", s).strip()


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def contains_match(pred: str, gold: str) -> bool:
    """Primary accuracy metric: normalised gold answer appears as whole words in prediction."""
    p, g = normalize(pred), normalize(gold)
    if not g:
        return False
    if g in ("yes", "no"):            # avoid "no" matching inside "not enough information"
        first = p.split(" ")[0] if p else ""
        return first == g
    return re.search(rf"(?<!\w){re.escape(g)}(?!\w)", p) is not None


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if not p or not g or n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)
