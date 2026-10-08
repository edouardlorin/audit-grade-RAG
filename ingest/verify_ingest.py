"""Post-ingest coverage report: what got indexed, and what silently did not.

Never trust an ingest you have not inspected. This answers "did I produce
chunks"; eval/xml_coverage.py answers the harder question, "did I find the
clauses that actually exist".

Usage:
    python verify_ingest.py
"""
from __future__ import annotations

import collections
import os
import sys

from qdrant_client import QdrantClient

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = os.getenv("RAG_COLLECTION", "standards_v1")

# Acceptance criteria (see the install guide).
MAX_NO_CLAUSE_PCT = 10.0
MAX_NO_PAGE_PCT = 2.0
MIN_MEDIAN_CHARS = 800
MAX_MEDIAN_CHARS = 2400

try:
    QDRANT_KEY = os.environ["QDRANT_API_KEY"]
except KeyError:
    sys.exit("QDRANT_API_KEY is not set. In PowerShell:\n"
             "  $env:QDRANT_API_KEY = ((Get-Content C:\\rag\\qdrant\\.env) "
             "-replace 'QDRANT_API_KEY=','')")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY, timeout=60.0)

if not client.collection_exists(COLLECTION):
    sys.exit(f"collection '{COLLECTION}' does not exist - run index.py first")

per_doc = collections.Counter()
no_page = collections.Counter()
no_clause = collections.Counter()
pdf_pages = collections.defaultdict(set)
ctypes = collections.Counter()
lengths = []

offset = None
while True:
    points, offset = client.scroll(
        COLLECTION, limit=512, offset=offset,
        with_payload=True, with_vectors=False)
    if not points:
        break
    for p in points:
        pl = p.payload
        doc = pl.get("doc_id", "?")
        per_doc[doc] += 1
        if pl.get("page_start") is None:
            no_page[doc] += 1
        if pl.get("clause_no") in (None, "", "0"):
            no_clause[doc] += 1
        pdf_pages[doc].add(pl.get("pdf_page_start"))
        ctypes[pl.get("content_type", "?")] += 1
        lengths.append(len(pl.get("body", "")))
    if offset is None:
        break

if not per_doc:
    sys.exit(f"collection '{COLLECTION}' is empty")

print(f"{'doc_id':<28}{'chunks':>8}{'no page':>10}{'no clause':>11}"
      f"{'pages':>8}{'chunks/pg':>11}")
print("-" * 76)

failures = []
for doc, n in per_doc.most_common():
    npg = len(pdf_pages[doc])
    clause_pct = 100 * no_clause[doc] / n
    page_pct = 100 * no_page[doc] / n
    print(f"{doc:<28}{n:>8}{no_page[doc]:>9} {page_pct:>4.0f}%"
          f"{no_clause[doc]:>7} {clause_pct:>3.0f}%{npg:>8}"
          f"{n / max(npg, 1):>11.1f}")
    if clause_pct > MAX_NO_CLAUSE_PCT:
        failures.append(f"{doc}: {clause_pct:.0f}% of chunks have no clause "
                        f"number (limit {MAX_NO_CLAUSE_PCT:.0f}%) - heading "
                        f"detection is failing for this publisher")
    if page_pct > MAX_NO_PAGE_PCT:
        failures.append(f"{doc}: {page_pct:.0f}% of chunks have no printed "
                        f"page (limit {MAX_NO_PAGE_PCT:.0f}%) - check "
                        f"page_offset / page_map in registry.yaml")

lengths.sort()
if lengths:
    median = lengths[len(lengths) // 2]
    p90 = lengths[int(len(lengths) * 0.9)]
    print(f"\nchunk body chars   p50={median}  p90={p90}  max={lengths[-1]}")
    if not MIN_MEDIAN_CHARS <= median <= MAX_MEDIAN_CHARS:
        failures.append(
            f"median chunk length {median} outside {MIN_MEDIAN_CHARS}-"
            f"{MAX_MEDIAN_CHARS}: too small means over-fragmentation, "
            f"too large means headings are being missed")

print("content types     " + "  ".join(f"{k}={v}" for k, v in ctypes.most_common()))

print()
if failures:
    print("FAILED acceptance criteria:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("all acceptance criteria met")
