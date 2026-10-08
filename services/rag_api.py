"""Orchestration: encode -> hybrid search -> rerank -> gate -> generate -> verify.

The LLM appears exactly once, at the end. It never embeds, and the reranker
never sees a generated answer: the reranker scores (query, candidate) pairs
BEFORE generation, which is what lets the abstention gate refuse without ever
invoking the model.

Run:
    uvicorn rag_api:app --host 127.0.0.1 --port 8080 --app-dir C:\\rag\\services

Depends on: retrieval-svc (8081), Qdrant (6333), Ollama (11434).
"""
from __future__ import annotations

import datetime
import json
import os
import pathlib
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient, models

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
RETRIEVAL = os.getenv("RETRIEVAL_URL", "http://127.0.0.1:8081")
OLLAMA = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.getenv("RAG_MODEL", "std-rag")
COLLECTION = os.getenv("RAG_COLLECTION", "standards_v1")

LOG = pathlib.Path(os.getenv("RAG_LOG", r"C:\rag\logs\queries.jsonl"))
PROMPT_PATH = pathlib.Path(
    os.getenv("RAG_PROMPT", r"C:\rag\services\prompts\answer_contract.txt"))
# The UI is served from this same origin so the browser does not treat its
# /ask call as cross-origin. Opening index.html from the filesystem instead
# would be blocked, and adding CORS to allow it would open the API to any page.
STATIC_DIR = pathlib.Path(os.getenv("RAG_STATIC", r"C:\rag\services\static"))

# --- retrieval shape ------------------------------------------------------
CANDIDATES = 50        # retrieved by hybrid search, then reranked
KEEP = 6               # passages shown to the LLM
MAX_PER_CLAUSE = 3     # stop one long clause consuming every slot
MAX_PER_DOC = 4

# --- abstention thresholds ------------------------------------------------
# Calibrate these against your golden set; see the tuning order in the guide.
# TAU_ABSTAIN decides whether to answer at all. TAU_INCLUDE decides which of
# the survivors are good enough to show the model - padding the context with
# weakly relevant clauses measurably increases citation errors.
TAU_ABSTAIN = float(os.getenv("TAU_ABSTAIN", "0.35"))
TAU_INCLUDE = float(os.getenv("TAU_INCLUDE", "0.15"))

NUM_CTX = int(os.getenv("RAG_NUM_CTX", "16384"))
TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.1"))
SEED = int(os.getenv("RAG_SEED", "42"))

MARKER_RE = re.compile(r"\[S(\d+)\]")
QUOTE_RE = re.compile(r"[\u201c\"]([^\u201d\"]{12,400})[\u201d\"]")

try:
    QDRANT_KEY = os.environ["QDRANT_API_KEY"]
except KeyError:
    sys.exit("QDRANT_API_KEY is not set")

if not PROMPT_PATH.exists():
    sys.exit(f"answer contract not found at {PROMPT_PATH}")
SYSTEM = PROMPT_PATH.read_text(encoding="utf-8")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY, timeout=60.0)
app = FastAPI(title="rag-api", version="1.1")


class Ask(BaseModel):
    question: str
    doc_filter: Optional[List[str]] = None


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

def _post(url: str, payload: dict, timeout: float = 300.0) -> dict:
    r = httpx.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def hybrid_search(dense: List[float], sparse: Dict[str, List],
                  limit: int = CANDIDATES,
                  doc_filter: Optional[List[str]] = None):
    """Dense + sparse prefetch fused by Reciprocal Rank Fusion, one round trip.

    RRF rather than a weighted score sum: cosine similarity and lexical scores
    are not on a common scale, so any fixed weighting would be wrong for some
    queries. RRF combines ranks and sidesteps calibration entirely.
    """
    flt = None
    if doc_filter:
        flt = models.Filter(must=[models.FieldCondition(
            key="doc_id", match=models.MatchAny(any=doc_filter))])

    res = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=dense, using="dense",
                            limit=limit * 2, filter=flt),
            models.Prefetch(
                query=models.SparseVector(indices=sparse["indices"],
                                          values=sparse["values"]),
                using="lexical", limit=limit * 2, filter=flt),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        query_filter=flt,
        with_payload=True,
    )
    return res.points


def diversify(ranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cap chunks per clause and per document so the context stays varied."""
    out, per_clause, per_doc = [], {}, {}
    for c in ranked:
        ck = (c.get("doc_id"), c.get("clause_no"))
        if per_clause.get(ck, 0) >= MAX_PER_CLAUSE:
            continue
        if per_doc.get(c.get("doc_id"), 0) >= MAX_PER_DOC:
            continue
        per_clause[ck] = per_clause.get(ck, 0) + 1
        per_doc[c.get("doc_id")] = per_doc.get(c.get("doc_id"), 0) + 1
        out.append(c)
        if len(out) >= KEEP:
            break
    return out


# --------------------------------------------------------------------------
# Citation rendering
# --------------------------------------------------------------------------

def page_ref(s: Dict[str, Any]) -> str:
    """Printed page if known, then the PDF's own label, then the PDF index."""
    if s.get("page_start") is not None:
        if s.get("page_end") and s["page_end"] != s["page_start"]:
            return f"pp. {s['page_start']}-{s['page_end']}"
        return f"p. {s['page_start']}"
    if s.get("page_label_start"):
        return f"p. {s['page_label_start']}"
    return f"PDF p. {s.get('pdf_page_start', '?')}"


def citation(s: Dict[str, Any]) -> str:
    return (f"{s.get('doc_id')} ({s.get('edition')}), "
            f"clause {s.get('clause_no')} \u2014 {s.get('clause_title')}, "
            f"{page_ref(s)}")


def render_sources(sources: List[Dict[str, Any]]) -> str:
    """Number the passages. The model sees [S1]..[Sn] and nothing else."""
    parts = []
    for i, s in enumerate(sources, start=1):
        parts.append(
            f"[S{i}] document={s.get('doc_id')} edition={s.get('edition')} "
            f"clause={s.get('clause_no')} "
            f"title=\"{s.get('clause_title')}\" page={page_ref(s)}\n"
            f"{s.get('body', '')}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def verify(answer: str, sources: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Deterministic post-generation check. Prompting alone is not enough."""
    problems: List[str] = []
    n = len(sources)

    # 1. every marker must be in range
    for i in sorted({int(m) for m in MARKER_RE.findall(answer)}):
        if not 1 <= i <= n:
            problems.append(f"marker [S{i}] does not exist (have S1..S{n})")

    # 2. every factual sentence must carry a marker
    for sent in re.split(r"(?<=[.!?])\s+", answer.strip()):
        s = sent.strip()
        if not s or s.startswith("NOT_IN_CORPUS"):
            continue
        s = s.lstrip("-*\u2022 ")
        if len(s) < 25:                       # fragments, list headers
            continue
        if not MARKER_RE.search(s):
            problems.append(f"unsourced sentence: {s[:90]}")

    # 3. every quotation must appear verbatim in a source body
    #    (bodies, not `text`: the synthetic header must not be quotable)
    bodies = [_norm(src.get("body", "")) for src in sources]
    for q in QUOTE_RE.findall(answer):
        if not any(_norm(q) in b for b in bodies):
            problems.append(f"quotation not found in any source: {q[:70]}")

    # 4. rule 7: the model must never write a literal page or clause reference
    for pat in (r"\bp\.\s?\d{1,4}\b", r"\u00a7\s?\d", r"\bpage\s+\d{1,4}\b"):
        if re.search(pat, answer, re.I):
            problems.append("model emitted a literal page or clause reference")
            break

    return (not problems), problems


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def generate(question: str, sources: List[Dict[str, Any]],
             corrections: Optional[List[str]] = None) -> str:
    user = f"SOURCES\n{render_sources(sources)}\n\nQUESTION\n{question}"
    if corrections:
        user += ("\n\nYour previous answer violated the rules:\n- "
                 + "\n- ".join(corrections)
                 + "\nRewrite it correctly.")

    out = _post(f"{OLLAMA}/api/chat", {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_ctx": NUM_CTX,
                    "seed": SEED},
    })
    return out["message"]["content"].strip()


# --------------------------------------------------------------------------
# Endpoint
# --------------------------------------------------------------------------

@app.post("/ask")
def ask(req: Ask) -> Dict[str, Any]:
    t0 = time.perf_counter()

    enc = _post(f"{RETRIEVAL}/encode", {"texts": [req.question]})
    dense, sparse = enc["dense"][0], enc["sparse"][0]

    hits = hybrid_search(dense, sparse, CANDIDATES, req.doc_filter)
    if not hits:
        return _finish(req, {"status": "not_in_corpus",
                             "answer": "No passage in the corpus is relevant "
                                       "to this question.",
                             "sources": []}, t0)

    cands = [dict(h.payload) for h in hits]
    scores = _post(f"{RETRIEVAL}/rerank",
                   {"query": req.question,
                    "passages": [c.get("body", "") for c in cands]})["scores"]
    for c, s in zip(cands, scores):
        c["rerank"] = s
    cands.sort(key=lambda c: c["rerank"], reverse=True)

    # The abstention gate. Below threshold the LLM is never called, so there
    # is no opportunity to confabulate: control flow, not prompt wording.
    if cands[0]["rerank"] < TAU_ABSTAIN:
        return _finish(req, {
            "status": "not_in_corpus",
            "answer": "The corpus does not contain a passage that answers "
                      "this question.",
            "sources": [],
            "best_score": round(cands[0]["rerank"], 3),
        }, t0)

    sources = diversify([c for c in cands if c["rerank"] >= TAU_INCLUDE])

    answer = generate(req.question, sources)
    ok, problems = verify(answer, sources)
    if not ok:
        answer = generate(req.question, sources, corrections=problems)
        ok, problems = verify(answer, sources)

    if answer.startswith("NOT_IN_CORPUS"):
        result = {"status": "not_in_corpus", "answer": answer, "sources": []}
    elif not ok:
        result = {
            "status": "unverified",
            "answer": "The generated answer could not be verified against the "
                      "retrieved passages and has been withheld. The relevant "
                      "clauses are listed below for manual review.",
            "problems": problems,
            "sources": [_source_record(i + 1, s) for i, s in enumerate(sources)],
        }
    else:
        used = sorted({int(m) for m in MARKER_RE.findall(answer)})
        result = {
            "status": "answered",
            "answer": answer,
            "sources": [_source_record(i, sources[i - 1])
                        for i in used if 1 <= i <= len(sources)],
        }

    return _finish(req, result, t0)


def _source_record(marker: int, s: Dict[str, Any]) -> Dict[str, Any]:
    """Structured, so evaluation does not have to parse the display string."""
    return {
        "marker": f"[S{marker}]",
        "citation": citation(s),
        "doc_id": s.get("doc_id"),
        "clause_no": s.get("clause_no"),
        "page": s.get("page_start"),
        "rerank_score": round(s.get("rerank", 0.0), 3),
    }


def _finish(req: Ask, result: Dict[str, Any], t0: float) -> Dict[str, Any]:
    """Attach latency and append the audit trail line."""
    result["latency_s"] = round(time.perf_counter() - t0, 2)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "question": req.question,
            "doc_filter": req.doc_filter,
            "status": result["status"],
            "answer": result.get("answer", "")[:4000],
            "sources": result.get("sources", []),
            "best_score": result.get("best_score"),
            "latency_s": result["latency_s"],
            "model": MODEL,
            "collection": COLLECTION,
            "tau_abstain": TAU_ABSTAIN,
        }, ensure_ascii=False) + "\n")
    return result


@app.get("/", response_class=HTMLResponse)
def index():
    page = STATIC_DIR / "index.html"
    if not page.exists():
        return HTMLResponse(
            f"<p>No UI installed. Expected index.html at {STATIC_DIR}.</p>",
            status_code=404)
    return FileResponse(page, media_type="text/html")


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        pts = client.get_collection(COLLECTION).points_count
    except Exception as e:
        return {"status": "degraded", "qdrant_error": str(e)}
    return {"status": "ok", "collection": COLLECTION, "points": pts,
            "model": MODEL, "tau_abstain": TAU_ABSTAIN,
            "tau_include": TAU_INCLUDE}
