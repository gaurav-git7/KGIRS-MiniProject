"""Chunking strategies. All sizes are measured in tokens of the embedder's own tokenizer
(BGE-small, uncased WordPiece), and all chunks are slices of the original text.

    whole_doc : baseline, no chunking (the model truncates at 512 tokens)
    fixed     : consecutive windows of `size` tokens (optional overlap)
    recursive : split on paragraph, line, sentence, clause, word boundaries until pieces fit,
                then merge neighbouring pieces up to `size`
    semantic  : sentences grouped until adjacent-sentence similarity drops (corpus-wide
                percentile threshold) or `size` is reached
"""
import re
import time

import numpy as np

from . import config as C


class Tok:
    """Thin wrapper around the embedder tokenizer."""

    def __init__(self, name=C.EMBED_MODEL):
        from transformers import AutoTokenizer
        self.t = AutoTokenizer.from_pretrained(name)
        self.t.model_max_length = 10 ** 9           # we chunk long docs ourselves; silence length warnings

    def count(self, text):
        return len(self.t(text, add_special_tokens=False)["input_ids"])

    def counts(self, texts):
        if not texts:
            return []
        enc = self.t(texts, add_special_tokens=False)["input_ids"]
        return [len(x) for x in enc]

    def offsets(self, text):
        enc = self.t(text, add_special_tokens=False, return_offsets_mapping=True)
        return enc["offset_mapping"]


# ---- fixed-length ----------------------------------------------------------
def chunk_fixed(text, size, tok, overlap=0):
    off = tok.offsets(text)
    if not off:
        return []
    step = max(size - overlap, 1)
    out, i = [], 0
    while i < len(off):
        j = min(i + size, len(off))
        piece = text[off[i][0]:off[j - 1][1]].strip()
        if piece:
            out.append(piece)
        if j == len(off):
            break
        i += step
    return out


# ---- recursive -------------------------------------------------------------
SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


def chunk_recursive(text, size, tok, overlap=0):
    def split(t, seps):
        if tok.count(t) <= size:
            return [t]
        if not seps:
            return chunk_fixed(t, size, tok)          # last resort: no separator left
        sep, rest = seps[0], seps[1:]
        parts = t.split(sep)
        if len(parts) == 1:
            return split(t, rest)
        pieces = []
        for i, p in enumerate(parts):
            p = p + (sep if i < len(parts) - 1 else "")      # keep the separator so text is preserved
            pieces.extend(split(p, rest) if tok.count(p) > size else [p])
        return pieces

    pieces = split(text, SEPARATORS)
    counts = tok.counts(pieces)
    chunks, cur, cur_n = [], "", 0
    for p, n in zip(pieces, counts):
        if cur and cur_n + n > size:
            chunks.append(cur)
            cur, cur_n = p, n
        else:
            cur, cur_n = cur + p, cur_n + n
    if cur:
        chunks.append(cur)
    return [c.strip() for c in chunks if c.strip()]


# ---- semantic --------------------------------------------------------------
_SENT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(\[])|\n+')


def split_sentences(text):
    return [s.strip() for s in _SENT.split(text) if s and s.strip()]


class SemanticState:
    """Sentence embeddings for one corpus, computed once and shared by both chunk sizes.
    The time it took is added to every semantic run's chunking time so cost stays honest."""

    def __init__(self, docs, tok, embedder):
        t0 = time.time()
        self.sents = {d: split_sentences(t) for d, t in docs.items()}
        flat = [s for d in self.sents for s in self.sents[d]]
        emb = embedder.encode_texts(flat)                      # (n_sent, dim) normalised, torch
        self.ntok = dict(zip(range(len(flat)), tok.counts(flat)))
        self.dist, pos = {}, 0
        all_d = []
        for d in self.sents:
            n = len(self.sents[d])
            e = emb[pos:pos + n]
            if n > 1:
                dd = (1 - (e[:-1] * e[1:]).sum(1)).cpu().numpy()
                all_d.append(dd)
            else:
                dd = np.zeros(0)
            self.dist[d] = (pos, dd)
            pos += n
        self.threshold = float(np.percentile(np.concatenate(all_d), C.SEMANTIC_PERCENTILE)) if all_d else 1.0
        self.seconds = time.time() - t0


def chunk_semantic(doc_id, state, size, tok, overlap=0):
    sents = state.sents[doc_id]
    start, dist = state.dist[doc_id]
    min_tokens = max(int(size * C.SEMANTIC_MIN_FRAC), 1)
    chunks, cur, cur_n = [], [], 0

    def flush():
        nonlocal cur, cur_n
        if cur:
            chunks.append(" ".join(cur))
        cur, cur_n = [], 0

    for i, s in enumerate(sents):
        n = state.ntok[start + i]
        if n > size:                                   # one sentence longer than a chunk: hard split
            flush()
            chunks.extend(chunk_fixed(s, size, tok))
            continue
        if cur and (cur_n + n > size or (i > 0 and dist[i - 1] > state.threshold and cur_n >= min_tokens)):
            flush()
        cur.append(s)
        cur_n += n
    flush()
    return chunks


# ---- driver ----------------------------------------------------------------
def build_chunks(corpus_docs, titles, strategy, size, tok, sem_state=None):
    """corpus_docs: {doc_id: text}. Returns (chunk_texts, chunk_doc_ids, seconds spent chunking)."""
    t0 = time.time()
    texts, owners = [], []
    for d, text in corpus_docs.items():
        if strategy == "whole_doc":
            pieces = [text]
        elif strategy == "fixed":
            pieces = chunk_fixed(text, size, tok, C.CHUNK_OVERLAP)
        elif strategy == "recursive":
            pieces = chunk_recursive(text, size, tok, C.CHUNK_OVERLAP)
        elif strategy == "semantic":
            pieces = chunk_semantic(d, sem_state, size, tok, C.CHUNK_OVERLAP)
        else:
            raise ValueError(strategy)
        if not pieces:
            pieces = [text or " "]
        if C.PREPEND_TITLE and strategy != "whole_doc" and titles.get(d):
            pieces = [p if i == 0 else f"{titles[d]}\n{p}" for i, p in enumerate(pieces)]
        texts.extend(pieces)
        owners.extend([d] * len(pieces))
    seconds = time.time() - t0
    if strategy == "semantic" and sem_state is not None:
        seconds += sem_state.seconds
    return texts, owners, seconds
