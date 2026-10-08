"""PyMuPDF parsing that preserves the structural signals a clause chunker needs.

Emits one record per layout block, tagged as heading / body / table, carrying the
page number in PDF coordinates, printed coordinates, and the PDF's own page label
where one exists.

Supported heading grammars:
  EASA CS / AMC / GM        CS 25.1309, AMC 25.571, GM1 25.1309, CS 25.807(a)
  EASA structural           Book 1, Subpart F, Appendix H
  EASA implementing rules   21.A.14, 145.A.30, M.A.301, AMC 21.A.14
  Generic decimal           7, 7.2, 7.2.3          (ECSS, ISO, IEC)
  Annex                     Annex A, Annexe B
  MIL-STD method            METHOD 514.8

Add a new publisher by writing one regex and inserting it into _classify ahead of
the generic decimal pattern. Order matters: most specific first.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import fitz  # PyMuPDF

PARSER_VERSION = "1.1"

# Changing this constant changes chunk boundaries, so it invalidates the index.
# See the re-index policy table in the install guide before touching it.

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------

# A heading must be at least this much larger than the median body text size,
# OR be bold. Lower toward 1.0 for publishers whose headings are bold-only.
PROMINENCE = 1.04

# Blocks longer than this are never headings, whatever they look like.
MAX_HEADING_CHARS = 200

# Blocks whose characters are more than this fraction dots are table-of-contents
# leader lines ("CS 25.1309 Equipment .......... 3-F-12"). CS-25 opens with many.
TOC_DOT_RATIO = 0.20

# Contents lines whose leader sits in a separate layout block arrive here with
# no dots at all, so the ratio test above misses them. This catches the tail:
# dots, a wide gap or an ellipsis, followed by a page reference - plain (12),
# roman (xii) or composite EASA style (1-G-9).
# Publishers with alphanumeric numbering (EASA) must NOT fall through to the
# generic decimal pattern. AMC clauses carry internal numbered sections
# ("1 Purpose", "2 Related regulations"), which CLAUSE_RE reads as new clauses:
# it manufactures phantom entries AND fragments the parent clause, because each
# false heading triggers a flush. False for EASA, True for ECSS/ISO/IEC.
ALLOW_GENERIC_DECIMAL = True

# Trailing sub-paragraph suffixes on an identifier: "(g)", "(h)(3)".
# EASA publishes AMCs aimed at specific sub-paragraphs, so the suffix is real
# citation precision and is KEPT in clause_no. clause_base() strips it for
# grouping, filtering and coverage comparison against the eRules XML, whose
# identifiers carry no suffix.
CLAUSE_SUFFIX_RE = re.compile(r"(?:\([a-z0-9]{1,3}\))+\s*$", re.I)

# Sub-paragraph markers: "(1)", "(a)", "(iii)", "1)", "a)". Structure INSIDE a
# clause; must never open a new one.
SUBPARA_RE = re.compile(
    r"^\s*\(?\s*(?:\d{1,3}|[a-z]{1,2}|[ivxlc]{1,6})\s*\)")

TOC_TAIL_RE = re.compile(
    r"(?:\.{3,}|\u2026|\s{4,})\s*"
    r"(?:\d{1,4}|[ivxlcIVXLC]{1,7}|\d{1,3}-[A-Z]{1,2}-\d{1,4})\s*$")

# --------------------------------------------------------------------------
# Heading grammars
# --------------------------------------------------------------------------

# Separator between an identifier and its title: space, en/em dash, colon, dot,
# hyphen, in any combination or none at all.
DASH = r"[\s\u2013\u2014:.\-]*"

# --- EASA structural dividers ---------------------------------------------
BOOK_RE = re.compile(
    rf"^\s*(BOOK\s+\d)\b{DASH}(.{{0,{MAX_HEADING_CHARS}}}?)\s*$", re.I)
SUBPART_RE = re.compile(
    rf"^\s*(SUBPART\s+[A-Z]{{1,2}})\b{DASH}(.{{0,{MAX_HEADING_CHARS}}}?)\s*$", re.I)
APPENDIX_RE = re.compile(
    rf"^\s*(APPENDIX\s+[A-Z]{{1,2}})\b{DASH}(.{{0,{MAX_HEADING_CHARS}}}?)\s*$", re.I)

# --- Appendix sub-sections: H25.1, H25.4, J25.2 ---------------------------
# Globally unique (only one H25.1 in CS-25), so these stand as clause_no in
# their own right rather than needing composition against the parent.
APPENDIX_SUB_RE = re.compile(
    rf"^\s*([A-Z]\d{{2}}\.\d{{1,3}})\b{DASH}(.{{0,{MAX_HEADING_CHARS}}}?)\s*$")

# --- EASA certification specifications ------------------------------------
# CS 25.1309 / AMC 25.571 / GM1 25.1309 / AMC1 25.1309 / CS 25.807(a)(1)
# NOTE the lookahead rather than \b: an identifier may end in ')', and \b after
# ')' followed by a space is not a boundary, so the engine would backtrack and
# drop the '(a)' suffix into the title.
EASA_RULE_RE = re.compile(
    rf"^\s*((?:CS|AMC|GM)\d?)\s+"
    rf"((?:\d{{1,3}}\.\d{{1,4}}[A-Z]?|\d{{2,3}}[A-Z]\d{{2,4}})"
    rf"(?:\([a-z0-9]{{1,3}}\))*)"
    rf"(?=[\s\u2013\u2014:.\-]|$){DASH}(.{{0,{MAX_HEADING_CHARS}}}?)\s*$")

# --- EASA implementing rules (Part-21, Part-M, Part-145, Part-CAMO, ...) ---
# 21.A.14 / 145.A.30 / M.A.301 / AMC 21.A.14
EASA_PART_RE = re.compile(
    rf"^\s*((?:AMC|GM)\d?\s+)?"
    rf"((?:\d{{2,3}}|M|ML|CAMO|CAO|SPA|ARO|ORO|NCC|NCO|SPO)\.[AB]\.\d{{1,4}})"
    rf"\b{DASH}(.{{0,{MAX_HEADING_CHARS}}}?)\s*$")

# --- generic decimal numbering (ECSS, ISO, IEC, EN) ------------------------
CLAUSE_RE = re.compile(
    rf"^\s*(?:(?:Clause|Section|Para(?:graph)?|\u00a7)\s*)?"
    rf"(\d{{1,2}}(?:\.\d{{1,3}}){{0,4}})\s+"
    rf"(\S.{{0,{MAX_HEADING_CHARS}}}?)\s*$")

ANNEX_RE = re.compile(
    rf"^\s*(Annex(?:e)?\s+[A-Z])\b{DASH}(.{{0,{MAX_HEADING_CHARS}}}?)\s*$", re.I)

# MIL-STD style
METHOD_RE = re.compile(
    rf"^\s*(METHOD\s+\d{{3}}(?:\.\d)?)\b{DASH}(.{{0,{MAX_HEADING_CHARS}}}?)\s*$", re.I)


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Block:
    kind: str                              # "heading" | "body" | "table"
    text: str
    pdf_page: int                          # 1-based index into the PDF
    printed_page: Optional[int]            # derived from page_offset / page_map
    printed_label: Optional[str] = None    # the PDF's own page label, e.g. "3-F-12"
    clause_no: Optional[str] = None
    clause_title: Optional[str] = None
    level: int = 0
    size: float = 0.0
    bbox: tuple = field(default_factory=tuple)


# --------------------------------------------------------------------------
# Page numbering
# --------------------------------------------------------------------------

def printed_page(pdf_page: int, offset, page_map) -> Optional[int]:
    """Map a 1-based PDF page index to the number printed on the page.

    page_map wins if present; it is a list of
        {"from_pdf": int, "to_pdf": int, "offset": int}
    for documents whose numbering restarts partway through.
    """
    if page_map:
        for rng in page_map:
            if rng["from_pdf"] <= pdf_page <= rng["to_pdf"]:
                return pdf_page + rng["offset"]
        return None
    if offset is None:
        return None
    return pdf_page + offset


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------

def clause_base(clause_no: Optional[str]) -> Optional[str]:
    """'AMC 25.101(h)(3)' -> 'AMC 25.101'. Idempotent; None-safe."""
    if not clause_no:
        return clause_no
    return CLAUSE_SUFFIX_RE.sub("", clause_no).strip()


def _body_size(doc, sample: int = 25) -> float:
    """Median span size across sampled pages: the body text size."""
    sizes: List[float] = []
    step = max(1, len(doc) // sample)
    for i in range(0, len(doc), step):
        d = doc[i].get_text("dict")
        for blk in d.get("blocks", []):
            for line in blk.get("lines", []):
                for span in line.get("spans", []):
                    if span["text"].strip():
                        sizes.append(round(span["size"], 1))
    return statistics.median(sizes) if sizes else 10.0


def _join_spans(line: dict, default_size: float = 10.0) -> str:
    """Assemble a line from its spans, inserting a space only where the
    geometry shows one.

    Small caps are rendered as a full-size initial followed by smaller
    capitals, which PyMuPDF returns as adjacent spans. Joining spans with a
    blank unconditionally turns "APPENDIX" into "A PPENDIX" and no heading
    pattern can match it. Comparing each span's left edge with the previous
    span's right edge distinguishes a real word gap from a font change at the
    same pen position.
    """
    out, prev_x1, prev_size = [], None, default_size
    for span in line.get("spans", []):
        txt = span["text"]
        if not txt.strip():
            continue
        x0, x1 = span["bbox"][0], span["bbox"][2]
        if out and prev_x1 is not None:
            gap = x0 - prev_x1
            if (gap > 0.22 * prev_size
                    and not out[-1].endswith(" ")
                    and not txt.startswith(" ")):
                out.append(" ")
        out.append(txt)
        prev_x1, prev_size = x1, span["size"]
    return "".join(out)


def _clean_title(title: str) -> str:
    """Strip leading separator junk left by odd font encodings.

    Some CS-25 fonts map the en-dash to a codepoint that decodes as a stray
    letter, so the title arrives as 'u INSTRUCTIONS FOR ...'.
    """
    return re.sub(r"^[^\w(]{0,3}\s*", "", title).strip()


def _is_toc_leader(text: str) -> bool:
    """Table-of-contents lines masquerade as headings."""
    if len(text) <= 12:
        return False
    if text.count(".") > len(text) * TOC_DOT_RATIO:
        return True
    return bool(TOC_TAIL_RE.search(text))


def _classify(text: str, size: float, base: float, bold: bool,
              allow_generic: bool = True) -> Optional[Tuple[str, str, int]]:
    """Return (clause_no, clause_title, level) if this block heads a clause.

    Patterns are tried most specific first; the first match wins. The publisher
    prefix stays INSIDE clause_no, because 'CS 25.1309' and 'AMC 25.1309' are
    different clauses. Dropping the prefix makes them collide on the same
    chunk UUID and one silently overwrites the other.
    """
    if len(text) > MAX_HEADING_CHARS:
        return None
    if not (size >= base * PROMINENCE or bold):
        return None
    if SUBPARA_RE.match(text):
        return None            # "(1)", "(a)": structure inside a clause

    # --- EASA structural dividers -----------------------------------------
    m = BOOK_RE.match(text)
    if m:
        return m.group(1).title(), m.group(2).strip(), 1
    m = SUBPART_RE.match(text)
    if m:
        return m.group(1).title(), m.group(2).strip(), 2
    m = APPENDIX_RE.match(text)
    if m:
        return m.group(1).title(), _clean_title(m.group(2)), 2
    m = APPENDIX_SUB_RE.match(text)
    if m:
        return m.group(1).upper(), _clean_title(m.group(2)), 3

    # --- EASA certification specifications ---------------------------------
    m = EASA_RULE_RE.match(text)
    if m:
        prefix, number, title = m.group(1).upper(), m.group(2), m.group(3)
        level = 2 + number.count(".") + number.count("(")
        return f"{prefix} {number}", _clean_title(title), level

    # --- EASA implementing rules -------------------------------------------
    m = EASA_PART_RE.match(text)
    if m:
        prefix = (m.group(1) or "").strip().upper()
        no = f"{prefix} {m.group(2)}".strip()
        return no, m.group(3).strip(), 3 if prefix else 2

    # --- generic decimal numbering (ECSS / ISO / IEC only) ------------------
    if allow_generic:
        m = CLAUSE_RE.match(text)
        if m:
            no = m.group(1)
            return no, m.group(2).strip(), no.count(".") + 1

    m = ANNEX_RE.match(text)
    if m:
        return " ".join(m.group(1).split()).title(), m.group(2).strip(), 1

    m = METHOD_RE.match(text)
    if m:
        return m.group(1).upper(), m.group(2).strip(), 1

    return None


def _table_to_markdown(tbl) -> str:
    """Serialise an extracted table so it survives chunking as one unit."""
    rows = tbl.extract()
    rows = [[("" if c is None else str(c)).replace("\n", " ").strip() for c in r]
            for r in rows if any(c for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |",
           "|" + "|".join(["---"] * width) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def parse_pdf(path: str, page_offset=None, page_map=None,
              skip_tables: bool = False,
              allow_generic_decimal: bool = ALLOW_GENERIC_DECIMAL
              ) -> List[Block]:
    """Parse a PDF into ordered, typed blocks.

    skip_tables=True disables find_tables(), which is the slowest stage by a
    wide margin (2-10 pages/s versus 30-80 for text). Use it for documents
    with no tables worth extracting.
    """
    doc = fitz.open(path)
    base = _body_size(doc)
    blocks: List[Block] = []

    for pno in range(len(doc)):
        page = doc[pno]
        pdf_page = pno + 1
        pp = printed_page(pdf_page, page_offset, page_map)
        try:
            label = page.get_label() or None
        except Exception:
            label = None

        # Tables first, so their rectangles can be masked out of prose.
        table_rects = []
        if not skip_tables:
            try:
                for tbl in page.find_tables().tables:
                    md = _table_to_markdown(tbl)
                    if md:
                        table_rects.append(fitz.Rect(tbl.bbox))
                        blocks.append(Block("table", md, pdf_page, pp,
                                            printed_label=label,
                                            bbox=tuple(tbl.bbox)))
            except Exception:
                # find_tables is heuristic and will occasionally raise on an
                # unusual layout. One bad page must never abort a 1000-page doc.
                pass

        d = page.get_text("dict")
        for blk in d.get("blocks", []):
            if blk.get("type") != 0:          # 0 = text, 1 = image
                continue

            rect = fitz.Rect(blk["bbox"])
            area = rect.get_area()
            if area and any(
                rect.intersects(tr) and (rect & tr).get_area() / area > 0.5
                for tr in table_rects
            ):
                continue                      # already captured as a table

            line_texts, sizes, bold = [], [], False
            for line in blk.get("lines", []):
                lt = _join_spans(line, base)
                if lt.strip():
                    line_texts.append(lt)
                for span in line.get("spans", []):
                    if span["text"].strip():
                        sizes.append(span["size"])
                        bold = bold or bool(span["flags"] & 2 ** 4)
            text = " ".join(" ".join(line_texts).split())

            if len(text) < 3:
                continue
            if _is_toc_leader(text):
                continue

            size = statistics.median(sizes) if sizes else base
            hit = _classify(text, size, base, bold,
                            allow_generic_decimal)

            if hit:
                no, title, level = hit
                blocks.append(Block("heading", text, pdf_page, pp,
                                    printed_label=label,
                                    clause_no=no, clause_title=title,
                                    level=level, size=size,
                                    bbox=tuple(blk["bbox"])))
            else:
                blocks.append(Block("body", text, pdf_page, pp,
                                    printed_label=label,
                                    size=size, bbox=tuple(blk["bbox"])))

    doc.close()
    return blocks


# --------------------------------------------------------------------------
# Self-test: python parse.py            -> check the regexes
#            python parse.py FILE.pdf   -> summarise a real document
# --------------------------------------------------------------------------

SELFTEST = [
    # (text, bold, expected clause_no or None)
    ("BOOK 1 \u2014 AIRWORTHINESS CODE",              True,  "Book 1"),
    ("SUBPART F \u2014 EQUIPMENT",                    True,  "Subpart F"),
    ("SUBPART FLIGHT",                                True,  None),
    ("Appendix H",                                    True,  "Appendix H"),
    ("CS 25.303 Factor of safety",                    True,  "CS 25.303"),
    ("CS 25.1309 Equipment, systems and installations", True, "CS 25.1309"),
    ("AMC 25.1309 System Design and Analysis",        True,  "AMC 25.1309"),
    ("AMC1 25.571 Damage Tolerance",                  True,  "AMC1 25.571"),
    ("GM1 25.1309 Guidance",                          True,  "GM1 25.1309"),
    ("CS 25.807(a) Passenger emergency exits",        True,  "CS 25.807(a)"),
    ("21.A.14 Production organisation approval",      True,  "21.A.14"),
    ("145.A.30 Personnel requirements",               True,  "145.A.30"),
    ("M.A.301 Continuing airworthiness tasks",        True,  "M.A.301"),
    ("AMC 21.A.14 Demonstration of capability",       True,  "AMC 21.A.14"),
    ("7.2.3 Random vibration",                        True,  "7.2.3"),
    ("Annex B Test specification",                    True,  "Annex B"),
    ("METHOD 514.8 Vibration",                        True,  "METHOD 514.8"),
    # --- must NOT match -------------------------------------------------
    ("as required by CS 25.1309, the applicant shall show that ...",
                                                      False, None),
    ("CS 25.1309 Equipment ................................ 3-F-12",
                                                      True,  None),  # TOC leader
    ("The following paragraph applies to all installations.",
                                                      True,  None),
    ("(1) Purpose",                                   True,  None),
    ("(a) General",                                   True,  None),
    ("(iii) Conditions",                              True,  None),
    ("1) Scope",                                      True,  None),
    ("APPENDIX H \u2013 INSTRUCTIONS FOR CONTINUED AIRWORTHINESS",
                                                      True,  "Appendix H"),
    ("H25.1 General",                                 True,  "H25.1"),
    ("H25.4 Airworthiness Limitations Section",       True,  "H25.4"),
    ("CS 25J901 Installation",                        True,  "CS 25J901"),
    ("AMC 25J901(c)(2) Assembly of Components",       True,  "AMC 25J901(c)(2)"),
]


def _selftest() -> int:
    fails = 0
    for text, bold, expect in SELFTEST:
        if _is_toc_leader(text):
            got = None
        else:
            hit = _classify(text, size=12.0, base=10.0, bold=bold)
            got = hit[0] if hit else None
        ok = (got == expect)
        fails += (not ok)
        print(f"  {'ok ' if ok else 'FAIL'}  {text[:52]:<54} -> "
              f"{got!r}" + ("" if ok else f"   expected {expect!r}"))
    print(f"\n{len(SELFTEST) - fails}/{len(SELFTEST)} passed")
    return fails


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit(_selftest())

    blocks = parse_pdf(sys.argv[1])
    heads = [b for b in blocks if b.kind == "heading"]
    tables = [b for b in blocks if b.kind == "table"]
    print(f"blocks {len(blocks)} | headings {len(heads)} | tables {len(tables)}")
    for b in heads[:15] + ([] if len(heads) < 25 else heads[-10:]):
        print(f"  p{b.pdf_page:>4} L{b.level}  {b.clause_no:<18} "
              f"{(b.clause_title or '')[:52]}")
