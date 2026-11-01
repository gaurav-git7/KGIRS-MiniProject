"""Minimal OpenAI-compatible /v1/embeddings server around sentence-transformers (used on Colab).

Serves BAAI/bge-small-en-v1.5 on :1236 so GraphRAG and the naive baseline can use the
same embedder they use locally via LM Studio.
"""
import os

import tiktoken
import uvicorn
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

MODEL = os.environ.get("EMBED_HF_MODEL", "BAAI/bge-small-en-v1.5")
PORT = int(os.environ.get("EMBED_PORT", "1236"))
model = SentenceTransformer(MODEL, device="cuda" if os.environ.get("EMBED_DEVICE", "cuda") == "cuda" else "cpu")
enc = tiktoken.get_encoding("cl100k_base")
app = FastAPI()


@app.get("/v1/models")
def models():
    return {"data": [{"id": MODEL, "object": "model"}]}


@app.post("/v1/embeddings")
def embeddings(body: dict):
    inp = body["input"]
    inp = [inp] if isinstance(inp, str) else inp
    vecs = model.encode([str(x) for x in inp], batch_size=64, normalize_embeddings=True)
    toks = sum(len(enc.encode(str(x), disallowed_special=())) for x in inp)
    return {"object": "list", "model": body.get("model", MODEL),
            "data": [{"object": "embedding", "index": i, "embedding": v.tolist()} for i, v in enumerate(vecs)],
            "usage": {"prompt_tokens": toks, "total_tokens": toks}}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
