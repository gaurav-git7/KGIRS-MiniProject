"""Token-counting proxy between GraphRAG and LM Studio.

GraphRAG does not surface per-query token usage through its API, so every
GraphRAG request goes through this proxy (default :1235 -> LM Studio :1234).
It forwards requests untouched and keeps running totals:

    GET  /_stats   -> cumulative counters (prompt/completion/embedding tokens, calls)
    POST /_reset   -> zero the counters

If LM Studio does not report `usage` (e.g. for streamed replies) the proxy
falls back to a tiktoken estimate and flags it in `estimated_calls`.

Run:  python scripts/usage_proxy.py
"""
import json
import sys
import threading
from pathlib import Path

import httpx
import tiktoken
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

UPSTREAM = C.LLM_BASE_URL.rstrip("/").removesuffix("/v1")
EMBED_UPSTREAM = C.EMBED_BASE_URL.rstrip("/").removesuffix("/v1")
PORT = 1235
ENC = tiktoken.get_encoding("cl100k_base")
LOCK = threading.Lock()


def _zero():
    return {"prompt_tokens": 0, "completion_tokens": 0, "embedding_tokens": 0,
            "llm_calls": 0, "embedding_calls": 0, "estimated_calls": 0}


STATS = _zero()
app = FastAPI()


def _count(text: str) -> int:
    return len(ENC.encode(text or "", disallowed_special=()))


def _messages_text(body: dict) -> str:
    msgs = body.get("messages") or []
    return "\n".join(m.get("content", "") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))
                     for m in msgs)


def _add(**kw):
    with LOCK:
        for k, v in kw.items():
            STATS[k] += v


@app.get("/_stats")
def stats():
    with LOCK:
        return dict(STATS)


@app.post("/_reset")
def reset():
    with LOCK:
        STATS.update(_zero())
    return {"ok": True}


@app.api_route("/{path:path}", methods=["GET", "POST"])
async def proxy(path: str, request: Request):
    url = f"{EMBED_UPSTREAM if path.endswith('embeddings') else UPSTREAM}/{path}"
    raw = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    body = {}
    if raw:
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {}
    is_chat = path.endswith("chat/completions")
    is_embed = path.endswith("embeddings")
    stream = bool(body.get("stream"))
    if is_chat and stream:
        body.setdefault("stream_options", {"include_usage": True})
        raw = json.dumps(body).encode()

    client = httpx.AsyncClient(timeout=None)

    if is_chat and stream:
        async def gen():
            text, usage = [], None
            async with client.stream("POST", url, content=raw, headers=headers) as r:
                async for line in r.aiter_lines():
                    yield (line + "\n").encode()
                    if line.startswith("data:") and "[DONE]" not in line:
                        try:
                            ev = json.loads(line[5:])
                        except json.JSONDecodeError:
                            continue
                        if ev.get("usage"):
                            usage = ev["usage"]
                        for ch in ev.get("choices", []):
                            text.append((ch.get("delta") or {}).get("content") or "")
            if usage:
                _add(prompt_tokens=usage.get("prompt_tokens", 0),
                     completion_tokens=usage.get("completion_tokens", 0), llm_calls=1)
            else:
                _add(prompt_tokens=_count(_messages_text(body)),
                     completion_tokens=_count("".join(text)), llm_calls=1, estimated_calls=1)
            await client.aclose()
        return StreamingResponse(gen(), media_type="text/event-stream")

    r = await client.request(request.method, url, content=raw or None, headers=headers)
    await client.aclose()
    try:
        data = r.json()
    except Exception:
        data = None
    if data is not None and r.status_code == 200:
        usage = data.get("usage") if isinstance(data, dict) else None
        if is_chat:
            if usage:
                _add(prompt_tokens=usage.get("prompt_tokens", 0),
                     completion_tokens=usage.get("completion_tokens", 0), llm_calls=1)
            else:
                out = "".join((c.get("message") or {}).get("content") or "" for c in data.get("choices", []))
                _add(prompt_tokens=_count(_messages_text(body)), completion_tokens=_count(out),
                     llm_calls=1, estimated_calls=1)
        elif is_embed:
            tok = (usage or {}).get("prompt_tokens") or (usage or {}).get("total_tokens")
            if not tok:
                inp = body.get("input", [])
                inp = [inp] if isinstance(inp, str) else inp
                tok = sum(_count(str(x)) for x in inp)
                _add(estimated_calls=1)
            _add(embedding_tokens=tok, embedding_calls=1)
        return JSONResponse(data, status_code=r.status_code)
    return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))


if __name__ == "__main__":
    print(f"usage proxy on :{PORT} -> {UPSTREAM}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
