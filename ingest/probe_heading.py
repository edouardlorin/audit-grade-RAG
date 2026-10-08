"""Find a region of the PDF and show why its headings did or did not classify.

Locates pages containing a search string, then dumps every text block on those
pages with its font size, bold flag, the filters it tripped, and the verdict
_classify() returned. Shows the raw text exactly as PyMuPDF assembled it, which
is usually where the surprise is.

Usage:
    python probe_heading.py "<pdf>" "Instructions for Continued Airworthiness"
    python probe_heading.py "<pdf>" "H25.1" --pages 2
    python probe_heading.py "<pdf>" "Appendix H" --all-blocks
"""
from __future__ import annotations

import argparse
import statistics
import sys

import fitz

import parse as P


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("needle", nargs="?", default=None,
                    help="text to search for; omit when using --page")
    ap.add_argument("--page", metavar="N[,N...]",
                    help="dump these 1-based PDF pages directly, no search")
    ap.add_argument("--min-page", type=int, default=0,
                    help="ignore search hits before this page (skips contents)")
    ap.add_argument("--pages", type=int, default=2,
                    help="how many matching pages to dump (default 2)")
    ap.add_argument("--all-blocks", action="store_true",
                    help="show body blocks too, not just heading candidates")
    ap.add_argument("--generic", action="store_true",
                    help="evaluate with allow_generic_decimal=True")
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    base = P._body_size(doc)
    print(f"median body size: {base:.2f}   "
          f"PROMINENCE={P.PROMINENCE} -> heading needs size >= "
          f"{base * P.PROMINENCE:.2f} or bold")

    if args.page:
        hits = [int(n) - 1 for n in args.page.split(",")]
        print(f"dumping pages {[h + 1 for h in hits]}\n")
    else:
        if not args.needle:
            sys.exit("give a search string, or use --page")
        hits = [i for i in range(len(doc))
                if args.needle.lower() in doc[i].get_text("text").lower()
                and i + 1 >= args.min_page]
        if not hits:
            sys.exit(f"'{args.needle}' not found at or after "
                     f"page {args.min_page}")
        print(f"found on {len(hits)} pages: {[h + 1 for h in hits[:14]]}"
              f"{' ...' if len(hits) > 14 else ''}")
        print("NOTE: early pages are usually the table of contents. Use "
              "--min-page to skip them, or --page to target one directly.\n")
        hits = hits[:args.pages]

    for pno in hits:
        page = doc[pno]
        print("=" * 100)
        print(f"PDF page {pno + 1}   label={page.get_label()!r}")
        print("=" * 100)

        d = page.get_text("dict")
        page_h = page.rect.height

        for blk in d.get("blocks", []):
            if blk.get("type") != 0:
                continue
            rect = fitz.Rect(blk["bbox"])
            parts, sizes, bold = [], [], False
            for line in blk.get("lines", []):
                for span in line.get("spans", []):
                    if span["text"].strip():
                        parts.append(span["text"])
                        sizes.append(span["size"])
                        bold = bold or bool(span["flags"] & 2 ** 4)
            text = " ".join(" ".join(parts).split())
            if len(text) < 3:
                continue
            size = statistics.median(sizes) if sizes else base

            prominent = size >= base * P.PROMINENCE or bold
            toc = P._is_toc_leader(text)
            hit = None if toc else P._classify(text, size, base, bold,
                                               args.generic)

            if not args.all_blocks and not prominent and not hit:
                continue

            flags = []
            if toc:
                flags.append("TOC-FILTERED")
            if not prominent:
                flags.append("not-prominent")
            if P.SUBPARA_RE.match(text):
                flags.append("SUBPARA-rejected")
            if rect.y1 < page_h * 0.06:
                flags.append("in-header-zone")
            if rect.y0 > page_h * 0.94:
                flags.append("in-footer-zone")

            verdict = (f"HEADING clause_no={hit[0]!r} level={hit[2]}"
                       if hit else "body")
            print(f"\n  y0={rect.y0:6.1f}  size={size:5.2f}  bold={bold!s:<5} "
                  f"-> {verdict}")
            if flags:
                print(f"     flags: {', '.join(flags)}")
            print(f"     text: {text[:160]!r}")

    doc.close()
    print("\nWhat to look for:")
    print("  - the heading text is not what you assumed (a prefix in front of")
    print("    'Appendix' breaks the ^ anchor in APPENDIX_RE)")
    print("  - the heading is split across two blocks")
    print("  - 'not-prominent': the heading is set in body type")
    print("  - 'TOC-FILTERED' on a real heading: the dot ratio is too tight")


if __name__ == "__main__":
    main()
