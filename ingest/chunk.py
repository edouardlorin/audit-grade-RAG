"""Assemble parsed blocks into clause-scoped chunks with full citation metadata.

A chunk is one clause, or a contiguous slice of one clause, and never spans two.
That constraint is what makes an audit-grade citation possible: every chunk can
be attributed to exactly one document / clause / page range.

Changing TARGET_CHARS, MAX_CHARS, OVERLAP_CHARS or the header format changes
chunk boundaries and therefore invalidates every vector already in Qdrant.
Re-index into a new collection; do not mix.
"""
from __future__ import annotations

import collections
import datetime
import hashlib
import re
import uuid
from dataclasses import dataclass
from typing import List, Optional

from parse import PARSER_VERSION, Block, clause_base

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------

TARGET_CHARS = 2400      # ~600 tokens: inside BGE-M3's window, big enough for a clause
MAX_CHARS = 4800         # hard ceiling before a forced split
OVERLAP_CHARS = 300      # carried between parts of a split clause

# A clause whose body is shorter than this is contents residue or a bare
# divider, not a citable requirement. Table-of-contents entries survive parsing
# as a heading plus a few characters of leader text, and land here.
MIN_BODY_CHARS = 100

# Fixed namespace so chunk IDs are reproducible across machines and runs.
# Never change this: it would orphan every point already stored.
NAMESPACE = uuid.UUID("6f1c9e2a-4b7d-4f10-9d3b-8a2e5c7d1f04")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    edition: str
    publisher: str
    lang: str
    clause_no: str
    clause_base: str          # clause_no without the (g)/(h)(3) suffix
    clause_title: str
    clause_path: str
    page_start: Optional[int]        # printed page number
    page_end: Optional[int]
    page_label_start: Optional[str]  # the PDF's own label, e.g. "3-F-12"
    page_label_end: Optional[str]
    pdf_page_start: int              # 1-based index into the PDF
    pdf_page_end: int
    content_type: str                # prose | table | mixed
    part: int
    n_parts: int
    text: str                        # header + body: this is what gets embedded
    body: str                        # body only: used for quote verification
    source_file: str
    source_sha256: str
    parser_version: str
    ingested_at: str


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

def _split_long(text: str, target: int, maximum: int,
                overlap: int) -> List[str]:
    """Split oversized clause text on paragraph then sentence boundaries.

    Never splits mid-word. Overlap is carried forward so a requirement that
    straddles a boundary is retrievable from either side.
    """
    if len(text) <= maximum:
        return [text]

    paras = [p.strip() for p in re.split(r"\n{2,}|(?<=\.)\s{2,}", text)
             if p.strip()]

    out: List[str] = []
    cur = ""
    for p in paras:
        if len(cur) + len(p) + 1 <= target or not cur:
            cur = f"{cur} {p}".strip()
        else:
            out.append(cur)
            tail = cur[-overlap:] if overlap else ""
            cur = f"{tail} {p}".strip()
    if cur:
        out.append(cur)

    # A single paragraph may still exceed the ceiling; split it hard.
    final: List[str] = []
    for seg in out:
        while len(seg) > maximum:
            cut = seg.rfind(" ", 0, maximum)
            cut = cut if cut > maximum // 2 else maximum
            final.append(seg[:cut])
            seg = seg[max(0, cut - overlap):]
        final.append(seg)

    return [s for s in final if s.strip()]


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def chunk_document(blocks: List[Block], meta: dict, sha256: str) -> List[Chunk]:
    """Walk parsed blocks, flushing a chunk each time a new clause begins."""
    stack: List[tuple] = []                    # [(clause_no, title, level)]
    cur_no, cur_title = "0", "(front matter)"
    buf: List[str] = []
    pages: List[tuple] = []                    # [(pdf_page, printed_page, label)]
    kinds: set = set()
    chunks: List[Chunk] = []
    seen_clauses: "collections.Counter[str]" = collections.Counter()
    dropped: List[tuple] = []
    now = datetime.datetime.now(
        datetime.timezone.utc).isoformat(timespec="seconds")

    def path_str() -> str:
        return " > ".join(f"{n} {t}".strip() for n, t, _ in stack) \
               or "(front matter)"

    def flush() -> None:
        nonlocal buf, pages, kinds
        body = "\n\n".join(b for b in buf if b.strip())
        if not body.strip() or not pages:
            buf, pages, kinds = [], [], set()
            return

        # Contents residue: a real clause number attached to almost no text.
        # Keeping these would put a citable-looking but empty chunk into the
        # index, competing in retrieval with the actual clause.
        if len(body.strip()) < MIN_BODY_CHARS and kinds != {"table"}:
            dropped.append((cur_no, pages[0][0], len(body.strip())))
            buf, pages, kinds = [], [], set()
            return

        # An identifier may legitimately recur (an Appendix restating a
        # requirement). The occurrence number keeps IDs distinct and stays
        # deterministic across runs of the same file.
        seen_clauses[cur_no] += 1
        occurrence = seen_clauses[cur_no]

        # The clause path is embedded WITH the body. The body of CS 25.1309
        # rarely repeats the string "25.1309", so without this a query naming
        # the clause number would not retrieve the clause.
        header = f"[{meta['doc_id']} {meta.get('edition', '')}] {path_str()}"
        parts = _split_long(body, TARGET_CHARS, MAX_CHARS, OVERLAP_CHARS)

        ctype = ("table" if kinds == {"table"}
                 else "mixed" if "table" in kinds else "prose")

        pdf_lo = min(p[0] for p in pages)
        pdf_hi = max(p[0] for p in pages)
        printed = [p[1] for p in pages if p[1] is not None]
        labels = [p[2] for p in pages if p[2]]

        for i, part_text in enumerate(parts, start=1):
            # Deterministic: re-ingesting updates in place. Includes edition, so
            # a new edition creates new points instead of overwriting the old.
            cid = str(uuid.uuid5(
                NAMESPACE,
                f"{meta['doc_id']}|{meta.get('edition', '')}|{cur_no}"
                f"|{occurrence}|{i}"))

            chunks.append(Chunk(
                chunk_id=cid,
                doc_id=meta["doc_id"],
                doc_title=meta.get("title", ""),
                edition=meta.get("edition", ""),
                publisher=meta.get("publisher", ""),
                lang=meta.get("lang", "en"),
                clause_no=cur_no,
                clause_base=clause_base(cur_no) or cur_no,
                clause_title=cur_title,
                clause_path=path_str(),
                page_start=min(printed) if printed else None,
                page_end=max(printed) if printed else None,
                page_label_start=labels[0] if labels else None,
                page_label_end=labels[-1] if labels else None,
                pdf_page_start=pdf_lo,
                pdf_page_end=pdf_hi,
                content_type=ctype,
                part=i,
                n_parts=len(parts),
                text=f"{header}\n\n{part_text}",
                body=part_text,
                source_file=meta["file"],
                source_sha256=sha256,
                parser_version=PARSER_VERSION,
                ingested_at=now,
            ))

        buf, pages, kinds = [], [], set()

    for b in blocks:
        if b.kind == "heading":
            # A heading repeating the identifier of the clause already open is
            # a running page header, not a new clause. Absorb it so a clause
            # spanning several pages stays one chunk group.
            if b.clause_no and b.clause_no == cur_no:
                pages.append((b.pdf_page, b.printed_page, b.printed_label))
                continue
            flush()
            # Pop siblings and deeper levels; keep genuine ancestors.
            while stack and stack[-1][2] >= b.level:
                stack.pop()
            stack.append((b.clause_no, b.clause_title, b.level))
            cur_no = b.clause_no or "0"
            cur_title = b.clause_title or ""
            pages.append((b.pdf_page, b.printed_page, b.printed_label))
        else:
            buf.append(b.text)
            kinds.add(b.kind)
            pages.append((b.pdf_page, b.printed_page, b.printed_label))

    flush()

    if dropped:
        print(f"    dropped {len(dropped)} short/contents entries "
              f"(e.g. {dropped[:3]})")
    return chunks


def sha256_file(path: str) -> str:
    """Hash the source PDF so a citation can be tied to an exact file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Self-test: python chunk.py
# --------------------------------------------------------------------------

if __name__ == "__main__":
    demo = [
        Block("heading", "Book 1", 1, 1, clause_no="Book 1",
              clause_title="Airworthiness Code", level=1),
        Block("heading", "Subpart F", 2, 2, clause_no="Subpart F",
              clause_title="Equipment", level=2),
        Block("heading", "CS 25.1309 Equipment", 3, 3, clause_no="CS 25.1309",
              clause_title="Equipment, systems and installations", level=3),
        Block("body", "The equipment shall be designed such that " * 12, 3, 3),
        Block("heading", "AMC 25.1309 System Design", 90, 90,
              clause_no="AMC 25.1309", clause_title="System Design", level=3),
        Block("body", "Acceptable means of compliance are as follows. " * 12,
              90, 90),
    ]
    meta = {"doc_id": "CS-25", "edition": "Amendment 27", "title": "CS-25",
            "publisher": "EASA", "lang": "en", "file": "raw/CS-25.pdf"}
    cs = chunk_document(demo, meta, "deadbeef")

    print(f"chunks: {len(cs)}   unique ids: {len({c.chunk_id for c in cs})}")
    for c in cs:
        print(f"  {c.clause_no:<14} p{c.page_start}  part {c.part}/{c.n_parts}"
              f"  {len(c.body):>5} chars")
        print(f"     path: {c.clause_path}")
    ids = {c.chunk_id for c in cs}
    assert len(ids) == len(cs), "chunk id collision"
    assert any(c.clause_no == "CS 25.1309" for c in cs)
    assert any(c.clause_no == "AMC 25.1309" for c in cs)
    print("\nok: CS and AMC produced distinct chunks with distinct ids")
