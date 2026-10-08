"""Compare the clauses you indexed against the official EASA eRules XML.

verify_ingest.py answers "did I produce chunks". This answers the harder
question: "did I find the clauses that actually exist". It is the only
objective measure of chunker quality available for CS-25.

The XML is NOT ingested. It is a measuring tape, used once, here.

Two expected differences, neither of which is a defect:
  - GM (Guidance Material) appears in the Easy Access XML but not in the
    standalone CS-25 PDF, which is Book 1 (CS) + Book 2 (AMC) only. GM is
    excluded from the miss count below.
  - The XML is unpaginated, so it cannot validate page numbers. Those still
    need a manual spot-check.

Usage:
    python xml_coverage.py "C:\\rag\\corpus\\CS-25 ... xml (machine).xml"
    python xml_coverage.py FILE.xml --doc-id CS-25
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import Set

from qdrant_client import QdrantClient

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = os.getenv("RAG_COLLECTION", "standards_v1")

# Schema-agnostic: regex the identifiers out of all text content, so this works
# whatever the eRules tag names turn out to be.
#
# This pattern does NOT capture a trailing "(g)" / "(h)(3)" sub-paragraph
# suffix, so the XML side is always base-form. The index keeps the suffix,
# because it is real citation precision. Both sides are therefore reduced to
# base form before comparison - comparing raw strings counts every
# "AMC 25.101(g)" as an identifier the XML lacks, and reports a phantom gap.
ID_RE = re.compile(r"\b((?:CS|AMC|GM)\d?)\s+(\d{1,3}\.\d{1,4}[A-Z]?)\b")

MISS_TARGET_PCT = 5.0

# Mirrors parse.clause_base(); duplicated so this script has no import
# dependency on the ingest package.
CLAUSE_SUFFIX_RE = re.compile(r"(?:\([a-z0-9]{1,3}\))+\s*$", re.I)


def xml_identifiers(path: str) -> Set[str]:
    root = ET.parse(path).getroot()
    found: Set[str] = set()
    for el in root.iter():
        for txt in (el.text, el.tail, *el.attrib.values()):
            for prefix, number in ID_RE.findall(txt or ""):
                found.add(f"{prefix.upper()} {number}")
    return found


def indexed_identifiers(doc_id: str) -> Set[str]:
    try:
        key = os.environ["QDRANT_API_KEY"]
    except KeyError:
        sys.exit("QDRANT_API_KEY is not set")

    client = QdrantClient(url=QDRANT_URL, api_key=key, timeout=60.0)
    if not client.collection_exists(COLLECTION):
        sys.exit(f"collection '{COLLECTION}' does not exist")

    found: Set[str] = set()
    raw: Set[str] = set()
    offset = None
    while True:
        pts, offset = client.scroll(
            COLLECTION, limit=512, offset=offset,
            with_payload=["clause_no", "clause_base", "doc_id"],
            with_vectors=False)
        if not pts:
            break
        for p in pts:
            if p.payload.get("doc_id") != doc_id:
                continue
            no = p.payload.get("clause_no")
            if not no:
                continue
            raw.add(no)
            # Prefer the stored base; fall back to stripping, so the script
            # still works against a collection indexed before clause_base
            # existed.
            found.add(p.payload.get("clause_base")
                      or CLAUSE_SUFFIX_RE.sub("", no).strip())
        if offset is None:
            break
    print(f"indexed identifiers: {len(raw)} raw -> {len(found)} base form")
    return found


def bucket(ids: Set[str]) -> str:
    counts = collections.Counter(i.split()[0] for i in ids if i)
    return "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("xml")
    ap.add_argument("--doc-id", default="CS-25")
    ap.add_argument("--show", type=int, default=25,
                    help="how many missed identifiers to list")
    args = ap.parse_args()

    xml_ids = xml_identifiers(args.xml)
    if not xml_ids:
        sys.exit("no CS/AMC/GM identifiers found in the XML - check the file")

    indexed = indexed_identifiers(args.doc_id)

    # GM is expected to be absent from the standalone PDF.
    comparable = {i for i in xml_ids if not i.startswith("GM")}
    missed = comparable - indexed
    extra = indexed - xml_ids

    miss_pct = 100 * len(missed) / max(len(comparable), 1)

    print(f"XML total        {len(xml_ids):>6}   {bucket(xml_ids)}")
    print(f"XML comparable   {len(comparable):>6}   (GM excluded)")
    print(f"indexed          {len(indexed):>6}   {bucket(indexed)}")
    print(f"missed           {len(missed):>6}   {miss_pct:.1f}%   "
          f"<- chunker gaps")
    print(f"extra            {len(extra):>6}   <- Book/Subpart/Appendix "
          f"headings, plus any over-matching")

    if missed:
        print(f"\nsample missed ({min(args.show, len(missed))} of {len(missed)}):")
        for i in sorted(missed)[:args.show]:
            print(f"  {i}")
    if extra:
        print(f"\nsample extra ({min(args.show, len(extra))} of {len(extra)}):")
        for i in sorted(extra)[:args.show]:
            print(f"  {i}")

    print()
    if miss_pct > MISS_TARGET_PCT:
        print(f"FAIL: {miss_pct:.1f}% missed exceeds the {MISS_TARGET_PCT:.0f}% "
              f"target. Inspect the sample above for a pattern - a whole "
              f"Subpart missing points at heading detection, scattered misses "
              f"at the prominence threshold in parse.py")
        sys.exit(1)
    print(f"PASS: {miss_pct:.1f}% missed, within the "
          f"{MISS_TARGET_PCT:.0f}% target")


if __name__ == "__main__":
    main()
