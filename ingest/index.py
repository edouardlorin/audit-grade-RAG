"""Ingestion driver: parse -> chunk -> embed -> upsert into Qdrant.

Reads the corpus register, processes every document listed, and stores one point
per chunk carrying both a dense and a sparse vector plus full citation metadata.

Prerequisites (both must be running):
    retrieval-svc   http://127.0.0.1:8081   BGE-M3 encoder
    Qdrant          http://127.0.0.1:6333

Usage:
    set QDRANT_API_KEY first, then
    python index.py C:\\rag\\corpus\\registry.yaml
    python index.py C:\\rag\\corpus\\registry.yaml --only CS-25
    python index.py C:\\rag\\corpus\\registry.yaml --dry-run

Re-running is safe. Chunk IDs are deterministic UUIDv5 values derived from
doc_id + edition + clause_no + part, so a second run updates points in place
rather than duplicating them. A new edition produces new IDs and coexists with
the old one; superseded editions are never silently overwritten.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from dataclasses import asdict
from typing import Dict, List

import httpx
import yaml
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from parse import parse_pdf
from chunk import Chunk, chunk_document, sha256_file

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
RETRIEVAL = os.getenv("RETRIEVAL_URL", "http://127.0.0.1:8081")
COLLECTION = os.getenv("RAG_COLLECTION", "standards_v1")

DENSE_DIM = 1024        # BGE-M3 output dimension. Changing the embedding model
                        # invalidates every existing vector - re-index into a
                        # new collection, do not mix.

EMBED_BATCH = 16        # chunks per /encode call; lower if the GPU is tight

try:
    QDRANT_KEY = os.environ["QDRANT_API_KEY"]
except KeyError:
    sys.exit("QDRANT_API_KEY is not set. In PowerShell:\n"
             "  $env:QDRANT_API_KEY = ((Get-Content C:\\rag\\qdrant\\.env) "
             "-replace 'QDRANT_API_KEY=','')")


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------

def ensure_collection(client: QdrantClient) -> None:
    """Create the collection and its payload indexes if absent. Idempotent."""
    if client.collection_exists(COLLECTION):
        return

    print(f"creating collection '{COLLECTION}'")
    client.create_collection(
        collection_name=COLLECTION,
        # Named vectors: one point carries both representations, which is what
        # makes the RRF hybrid query at retrieval time possible.
        vectors_config={
            "dense": models.VectorParams(
                size=DENSE_DIM,
                distance=models.Distance.COSINE,
                on_disk=False,
            )
        },
        sparse_vectors_config={
            "lexical": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        },
        hnsw_config=models.HnswConfigDiff(m=32, ef_construct=256),
        optimizers_config=models.OptimizersConfigDiff(default_segment_number=2),
    )

    # Payload indexes must exist for filtered search to avoid a full scan.
    for field_name, schema in [
        ("doc_id", models.PayloadSchemaType.KEYWORD),
        ("edition", models.PayloadSchemaType.KEYWORD),
        ("clause_no", models.PayloadSchemaType.KEYWORD),
        ("clause_base", models.PayloadSchemaType.KEYWORD),
        ("content_type", models.PayloadSchemaType.KEYWORD),
        ("lang", models.PayloadSchemaType.KEYWORD),
        ("page_start", models.PayloadSchemaType.INTEGER),
    ]:
        client.create_payload_index(COLLECTION, field_name=field_name,
                                    field_schema=schema)


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------

def embed(texts: List[str]) -> Dict:
    """Call retrieval-svc for dense + sparse vectors."""
    try:
        r = httpx.post(f"{RETRIEVAL}/encode", json={"texts": texts}, timeout=300.0)
        r.raise_for_status()
    except httpx.ConnectError:
        sys.exit(f"cannot reach retrieval-svc at {RETRIEVAL}\n"
                 "start it, or check: curl.exe -s "
                 f"{RETRIEVAL}/health")
    return r.json()


def upsert(client: QdrantClient, chunks: List[Chunk]) -> None:
    """Embed and store chunks in batches."""
    for i in tqdm(range(0, len(chunks), EMBED_BATCH), desc="  embed+upsert",
                  unit="batch"):
        window = chunks[i:i + EMBED_BATCH]
        enc = embed([c.text for c in window])

        points = []
        for c, dense, sparse in zip(window, enc["dense"], enc["sparse"]):
            payload = asdict(c)
            payload.pop("chunk_id")          # the id lives on the point, not the payload
            points.append(models.PointStruct(
                id=c.chunk_id,
                vector={
                    "dense": dense,
                    "lexical": models.SparseVector(
                        indices=sparse["indices"],
                        values=sparse["values"],
                    ),
                },
                payload=payload,
            ))

        client.upsert(collection_name=COLLECTION, points=points, wait=False)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def process(meta: dict, corpus_root: pathlib.Path,
            client: QdrantClient, dry_run: bool) -> int:
    path = corpus_root / meta["file"]
    if not path.exists():
        print(f"  SKIP - file not found: {path}")
        return 0

    sha = sha256_file(str(path))
    blocks = parse_pdf(
        str(path),
        page_offset=meta.get("page_offset"),
        page_map=meta.get("page_map"),
        skip_tables=meta.get("skip_tables", False),
        # EASA documents must disable the generic decimal pattern; see parse.py
        allow_generic_decimal=meta.get("allow_generic_decimal", True),
    )
    headings = sum(1 for b in blocks if b.kind == "heading")
    tables = sum(1 for b in blocks if b.kind == "table")

    chunks = chunk_document(blocks, meta, sha)

    # Guard against the silent-overwrite failure: two clauses that collapse to
    # the same UUID would leave one of them absent with no error raised.
    unique = len({c.chunk_id for c in chunks})
    print(f"  {len(blocks)} blocks ({headings} headings, {tables} tables) "
          f"-> {len(chunks)} chunks")
    if unique != len(chunks):
        print(f"  ERROR chunk id collision: {len(chunks)} chunks but only "
              f"{unique} unique ids.")
        print("        A publisher prefix is probably being dropped from "
              "clause_no in parse.py")
        return 0

    no_clause = sum(1 for c in chunks if c.clause_no in (None, "", "0"))
    no_page = sum(1 for c in chunks if c.page_start is None)
    print(f"  no clause: {no_clause} ({no_clause / max(len(chunks),1):.0%})  "
          f"no page: {no_page} ({no_page / max(len(chunks),1):.0%})")

    if dry_run:
        sample = chunks[len(chunks) // 2]
        print(f"  --- sample chunk ---\n  clause: {sample.clause_no}\n"
              f"  path  : {sample.clause_path}\n"
              f"  pages : {sample.page_start}-{sample.page_end}\n"
              f"  text  : {sample.text[:300]}...")
        return len(chunks)

    upsert(client, chunks)
    return len(chunks)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest a standards corpus.")
    ap.add_argument("registry", nargs="?",
                    default=r"C:\rag\corpus\registry.yaml")
    ap.add_argument("--only", metavar="DOC_ID",
                    help="process a single document from the register")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse and chunk, print stats, write nothing")
    args = ap.parse_args()

    registry_path = pathlib.Path(args.registry)
    reg = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    corpus_root = registry_path.parent

    docs = reg["documents"]
    if args.only:
        docs = [d for d in docs if d["doc_id"] == args.only]
        if not docs:
            sys.exit(f"no document with doc_id '{args.only}' in {registry_path}")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY, timeout=120.0)
    if not args.dry_run:
        ensure_collection(client)

    total = 0
    for meta in docs:
        print(f"\n=== {meta['doc_id']} ({meta.get('edition', '?')}) "
              f":: {meta['file']}")
        total += process(meta, corpus_root, client, args.dry_run)

    if args.dry_run:
        print(f"\ndry run complete: {total} chunks, nothing written")
        return

    # Raise the indexing threshold now that the bulk load is finished.
    client.update_collection(
        collection_name=COLLECTION,
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
    )
    count = client.get_collection(COLLECTION).points_count
    print(f"\ndone: {total} chunks written, collection now holds {count} points")


if __name__ == "__main__":
    main()
