"""Report text coverage per PDF so scanned documents are caught before indexing.

PyMuPDF returns an empty string for image-only pages. Without this check those
documents index silently as nothing at all.

Usage:
    python triage.py                          # scans C:\\rag\\corpus\\raw
    python triage.py C:\\path\\to\\folder
    python triage.py C:\\path\\to\\file.pdf
"""
from __future__ import annotations

import pathlib
import sys

import fitz  # PyMuPDF

# A page with fewer characters than this is header/footer only, or blank.
MIN_CHARS_PER_PAGE = 40

# Above this fraction of empty pages, the document needs OCR before ingestion.
OCR_THRESHOLD_PCT = 15.0


def triage(path: pathlib.Path) -> dict:
    doc = fitz.open(path)
    pages = len(doc)
    empty = 0
    chars = 0
    labels = 0

    for page in doc:
        t = page.get_text("text").strip()
        chars += len(t)
        if len(t) < MIN_CHARS_PER_PAGE:
            empty += 1
        try:
            if page.get_label():
                labels += 1
        except Exception:
            pass

    doc.close()
    return {
        "file": path.name,
        "pages": pages,
        "empty_pages": empty,
        "empty_pct": round(100 * empty / max(pages, 1), 1),
        "chars_per_page": round(chars / max(pages, 1)),
        "has_labels": labels > 0,
    }


def main() -> None:
    target = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                          else r"C:\rag\corpus\raw")

    if target.is_file():
        files = [target]
    else:
        files = sorted(target.glob("*.pdf"))

    if not files:
        sys.exit(f"no PDFs found at {target}")

    print(f"{'file':<50}{'pages':>7}{'empty':>8}{'chars/pg':>10}"
          f"{'labels':>8}  status")
    print("-" * 92)

    for p in files:
        r = triage(p)
        status = "OCR REQUIRED" if r["empty_pct"] > OCR_THRESHOLD_PCT else "ok"
        print(f"{r['file'][:48]:<50}{r['pages']:>7}{r['empty_pct']:>7}%"
              f"{r['chars_per_page']:>10}{str(r['has_labels']):>8}  {status}")

    print("\nlabels=True means the PDF carries its own page labels; prefer them "
          "over an integer page_offset when the numbering is composite "
          "(e.g. '3-F-12').")


if __name__ == "__main__":
    main()
