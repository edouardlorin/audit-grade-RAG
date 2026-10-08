"""BGE-M3 encoder + bge-reranker-v2-m3 cross-encoder, served over HTTP.

Holds roughly 2.8 GB of FP16 weights resident on the GPU. Start once and leave
running: model load takes 20-40 seconds and must not sit in your edit-test loop,
which is why this is a separate process from rag_api.py.

Run:
    uvicorn retrieval_svc:app --host 127.0.0.1 --port 8081 --app-dir C:\\rag\\services

Environment:
    RETRIEVAL_DEVICE   cuda (default) or cpu. Set cpu for Configuration B,
                       which frees ~2.8 GB of VRAM for a larger LLM quant at
                       the cost of 3-6 s per rerank.
    EMBED_MAX_LEN      default 1024 tokens. Raising it costs latency and VRAM;
                       CHANGING IT INVALIDATES THE INDEX.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from FlagEmbedding import BGEM3FlagModel, FlagReranker

DEVICE = os.getenv("RETRIEVAL_DEVICE", "cuda")
USE_FP16 = DEVICE == "cuda"          # Turing has no BF16 path; FP16 is required
MAX_LEN = int(os.getenv("EMBED_MAX_LEN", "1024"))

# Keep small. With the LLM already resident on a 16 GB card, a large batch of
# long passages will trip an OOM at exactly the wrong moment.
ENCODE_BATCH = int(os.getenv("ENCODE_BATCH", "8"))
RERANK_BATCH = int(os.getenv("RERANK_BATCH", "8"))

# Pin these in production. A repository can be updated in place, and a silent
# weight change under a stable name is a failure you cannot debug from symptoms.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

app = FastAPI(title="retrieval-svc", version="1.1")

print(f"loading {EMBED_MODEL} on {DEVICE} (fp16={USE_FP16}) ...", flush=True)
embedder = BGEM3FlagModel(EMBED_MODEL, use_fp16=USE_FP16, devices=DEVICE)
print(f"loading {RERANK_MODEL} on {DEVICE} ...", flush=True)
reranker = FlagReranker(RERANK_MODEL, use_fp16=USE_FP16, devices=DEVICE)
print("retrieval-svc ready", flush=True)


class EncodeRequest(BaseModel):
    texts: List[str]
    max_length: int = Field(default=MAX_LEN)


class RerankRequest(BaseModel):
    query: str
    passages: List[str]


def _sparse_to_qdrant(weights: Dict[str, float]) -> Dict[str, List]:
    """FlagEmbedding returns {token_id_as_str: weight}; Qdrant wants two lists."""
    items = [(int(k), float(v)) for k, v in weights.items() if float(v) > 0.0]
    items.sort(key=lambda kv: kv[0])
    return {"indices": [i for i, _ in items],
            "values": [v for _, v in items]}


@app.post("/encode")
def encode(req: EncodeRequest) -> Dict[str, Any]:
    """Dense + sparse vectors in one forward pass.

    The sparse channel is what makes exact identifier matching work:
    'CS 25.1309', 'Category 3', 'IP67'. Dense embeddings are systematically
    weak at exact-token matching, which is most of what a standards query is.
    """
    out = embedder.encode(
        req.texts,
        batch_size=ENCODE_BATCH,
        max_length=req.max_length,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    return {
        "dense": [v.tolist() for v in out["dense_vecs"]],
        "sparse": [_sparse_to_qdrant(w) for w in out["lexical_weights"]],
        "dim": len(out["dense_vecs"][0]),
    }


@app.post("/rerank")
def rerank(req: RerankRequest) -> Dict[str, Any]:
    """Cross-encode (query, passage) pairs.

    normalize=True applies a sigmoid, giving 0..1 scores that are comparable
    across queries and can therefore be thresholded. Raw logits cannot: drop
    normalize and the abstention gate in rag_api.py stops working.
    """
    if not req.passages:
        return {"scores": []}

    pairs = [[req.query, p] for p in req.passages]
    scores = reranker.compute_score(pairs, batch_size=RERANK_BATCH,
                                    normalize=True)
    if isinstance(scores, float):
        scores = [scores]
    return {"scores": [float(s) for s in scores]}


@app.get("/health")
def health() -> Dict[str, Any]:
    alloc = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0
    return {
        "status": "ok",
        "device": DEVICE,
        "fp16": USE_FP16,
        "embed_model": EMBED_MODEL,
        "rerank_model": RERANK_MODEL,
        "max_length": MAX_LEN,
        "gpu_alloc_gb": round(alloc, 2),
    }
