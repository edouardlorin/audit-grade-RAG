"""Score the pipeline against golden.yaml.

Reports retrieval, citation and abstention separately, because they fail for
different reasons and are fixed in different places.

Clause matching is BASE FORM: gold "CS 25.803" matches a returned
"CS 25.803(c)", because the sub-paragraph suffix is extra precision, not a
different clause. The publisher prefix is still significant - "AMC 25.803" is
a different clause from "CS 25.803" and will not match it.

Usage:
    python run_eval.py golden.yaml
    python run_eval.py golden.yaml --answers        # print every answer
    python run_eval.py golden.yaml --answers --only q003
    python run_eval.py --self-test                  # check the matching rules
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import sys
import textwrap

import httpx
import yaml

API = "http://127.0.0.1:8080/ask"

# Mirrors parse.clause_base(): "AMC 25.101(h)(3)" -> "AMC 25.101"
CLAUSE_SUFFIX_RE = re.compile(r"(?:\([a-z0-9]{1,3}\))+\s*$", re.I)


# --------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------

class C:
    """ANSI codes, blanked when the terminal cannot render them."""
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    GREY = "\033[90m"
    BOLD = "\033[1m"
    OFF = "\033[0m"

    @classmethod
    def disable(cls) -> None:
        for k in ("BLUE", "YELLOW", "GREEN", "RED", "GREY", "BOLD", "OFF"):
            setattr(cls, k, "")


def enable_ansi() -> bool:
    """Windows consoles need virtual-terminal processing switched on."""
    if not sys.stdout.isatty():
        return False
    if os.name != "nt":
        return True
    try:
        import ctypes
        k = ctypes.windll.kernel32
        # ENABLE_PROCESSED_OUTPUT | ENABLE_WRAP_AT_EOL | ENABLE_VT_PROCESSING
        return bool(k.SetConsoleMode(k.GetStdHandle(-11), 7))
    except Exception:
        return False

# Deployment gates.
#
# citation_precision is deliberately NOT a gate. It divides gold hits by every
# source the model cited, so a question with one gold clause answered from
# three sources scores 33% however correct it is. It is reported as a
# diagnostic only. "gold clause cited" is the metric that means something.
GATES = {
    "gold_cited_pct": 80.0,
    "abstain_pct": 95.0,
    "fabricated": 0,
    "false_abstain_pct": 15.0,
}


def norm_clause(c: str | None) -> str:
    """Base form, case- and whitespace-insensitive."""
    if not c:
        return ""
    c = " ".join(str(c).split())
    return CLAUSE_SUFFIX_RE.sub("", c).strip().upper()


def clause_matches(gold: str, got: str) -> bool:
    return bool(gold) and norm_clause(gold) == norm_clause(got)


def _self_test() -> int:
    cases = [
        ("CS 25.803",   "CS 25.803(c)",     True,  "suffix is extra precision"),
        ("CS 25.803",   "CS 25.803",        True,  "exact"),
        ("AMC 25.101",  "AMC 25.101(h)(3)", True,  "two suffix groups"),
        ("CS 25.803",   "cs  25.803",       True,  "case and spacing"),
        ("CS 25.803",   "AMC 25.803",       False, "prefix is significant"),
        ("CS 25.803",   "CS 25.807",        False, "different clause"),
        ("Appendix H",  "Appendix H",       True,  "structural heading"),
        ("CS 25.803",   "",                 False, "empty"),
    ]
    bad = 0
    for gold, got, want, why in cases:
        ok = clause_matches(gold, got) == want
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {gold!r:<14} vs {got!r:<18} "
              f"-> {clause_matches(gold, got)!s:<5}  ({why})")
    print(f"\n{len(cases) - bad}/{len(cases)} passed")
    return bad


def verdict(q: dict, r: dict) -> tuple[bool, str]:
    """Did this question pass? Returns (ok, one-line reason)."""
    status = r.get("status")

    if q["expect"] == "not_in_corpus":
        if status == "not_in_corpus":
            return True, f"correctly abstained (best score {r.get('best_score')})"
        return False, f"expected abstention, got '{status}'"

    if status == "not_in_corpus":
        return False, f"false abstention (best score {r.get('best_score')})"
    if status == "unverified":
        return False, f"answer failed verification: {r.get('problems')}"

    srcs = r.get("sources", [])
    for idx, s in enumerate(srcs, start=1):
        if any(g["doc_id"] == s.get("doc_id")
               and clause_matches(g["clause_no"], s.get("clause_no"))
               for g in q.get("gold", [])):
            return True, f"gold clause cited at rank {idx}"

    got = [s.get("clause_no") for s in srcs]
    want = [g["clause_no"] for g in q.get("gold", [])]
    return False, f"gold not cited. wanted {want}, got {got}"


def status_line(ok: bool, reason: str) -> str:
    tag = (f"{C.GREEN}{C.BOLD}STATUS OK{C.OFF}" if ok
           else f"{C.RED}{C.BOLD}STATUS KO{C.OFF}")
    return f"{tag}  {C.GREY}{reason}{C.OFF}"


def print_answer(q: dict, r: dict, width: int = 96) -> None:
    """Full detail for manual review."""
    print("\n" + "=" * width)
    print(f"{q['id']}   expect={q['expect']}   status={r['status']}   "
          f"{r.get('latency_s', 0)}s")
    print("-" * width)
    wrapped = "\n   ".join(textwrap.wrap(" ".join(q["question"].split()),
                                         width - 3))
    print(f"{C.BLUE}Q: {wrapped}{C.OFF}")
    if q.get("gold"):
        g = ", ".join(f"{x['doc_id']} {x['clause_no']}" for x in q["gold"])
        print(f"{C.YELLOW}gold: {g}{C.OFF}")
    if q.get("note"):
        print("note: " + q["note"])
    print("-" * width)
    for line in (r.get("answer") or "").splitlines():
        print("\n".join(textwrap.wrap(line, width)) if line.strip() else "")
    if r.get("best_score") is not None:
        print(f"\n[best rerank score: {r['best_score']}]")
    if r.get("problems"):
        print("\nverifier problems:")
        for p in r["problems"]:
            print(f"  - {p}")
    if r.get("sources"):
        print("\nsources returned:")
        for s in r["sources"]:
            hit = ""
            if q.get("gold"):
                hit = "  <== GOLD" if any(
                    clause_matches(g["clause_no"], s.get("clause_no"))
                    and g["doc_id"] == s.get("doc_id")
                    for g in q["gold"]) else ""
            colour = C.YELLOW if hit else ""
            off = C.OFF if hit else ""
            print(f"  {colour}{s.get('marker',''):<6} "
                  f"{s.get('rerank_score'):<6} "
                  f"{s.get('citation','')}{hit}{off}")
    ok, reason = verdict(q, r)
    print("-" * width)
    print(status_line(ok, reason))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("golden", nargs="?", default=r"C:\rag\eval\golden.yaml")
    ap.add_argument("--api", default=API)
    ap.add_argument("--answers", action="store_true",
                    help="print every answer in full for manual review")
    ap.add_argument("--only", metavar="ID", help="run a single question")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-color", action="store_true",
                    help="plain output, e.g. when redirecting to a file")
    args = ap.parse_args()

    if args.no_color or not enable_ansi():
        C.disable()

    if args.self_test:
        sys.exit(_self_test())

    gold = yaml.safe_load(open(args.golden, encoding="utf-8"))["questions"]
    if args.only:
        gold = [q for q in gold if q["id"] == args.only]
        if not gold:
            sys.exit(f"no question with id {args.only}")

    stats = collections.Counter()
    lat, cite_hits, cite_total, gold_ranks = [], 0, 0, []

    for q in gold:
        try:
            r = httpx.post(args.api, json={"question": q["question"]},
                           timeout=300.0).json()
        except httpx.HTTPError as e:
            print(f"  [E] {q['id']} request failed: {e}")
            stats["errors"] += 1
            continue

        lat.append(r.get("latency_s", 0.0))
        status = r["status"]
        ok, reason = verdict(q, r)
        if args.answers:
            print_answer(q, r)
        else:
            print(f"  [{q['id']}] {status_line(ok, reason)}")

        # ---- out-of-corpus ------------------------------------------
        if q["expect"] == "not_in_corpus":
            stats["neg_total"] += 1
            if status == "not_in_corpus":
                stats["neg_correct"] += 1
            else:
                stats["neg_FAIL"] += 1
            continue

        # ---- in-corpus ----------------------------------------------
        stats["pos_total"] += 1
        if status == "not_in_corpus":
            stats["pos_false_abstain"] += 1
            continue
        if status == "unverified":
            stats["pos_unverified"] += 1
            continue

        stats["pos_answered"] += 1
        srcs = r.get("sources", [])
        cite_total += len(srcs)

        hit_rank = None
        for idx, s in enumerate(srcs, start=1):
            if any(g["doc_id"] == s.get("doc_id")
                   and clause_matches(g["clause_no"], s.get("clause_no"))
                   for g in q["gold"]):
                cite_hits += 1
                if hit_rank is None:
                    hit_rank = idx

        if hit_rank:
            stats["pos_gold_cited"] += 1
            gold_ranks.append(hit_rank)

    # ---- report ------------------------------------------------------
    p = max(stats["pos_total"], 1)
    n = max(stats["neg_total"], 1)
    gold_pct = 100 * stats["pos_gold_cited"] / p
    abstain_pct = 100 * stats["neg_correct"] / n
    false_pct = 100 * stats["pos_false_abstain"] / p
    precision = 100 * cite_hits / max(cite_total, 1)

    print(f"\n--- in-corpus (n={stats['pos_total']})")
    print(f"  answered              {stats['pos_answered']:>4}")
    print(f"  gold clause cited     {stats['pos_gold_cited']:>4}  {gold_pct:>5.0f}%")
    print(f"  false abstentions     {stats['pos_false_abstain']:>4}  {false_pct:>5.0f}%")
    print(f"  unverified            {stats['pos_unverified']:>4}")
    if gold_ranks:
        gold_ranks.sort()
        print(f"  gold cited at rank    median "
              f"{gold_ranks[len(gold_ranks)//2]}  worst {gold_ranks[-1]}")
    print(f"--- out-of-corpus (n={stats['neg_total']})")
    print(f"  correctly abstained   {stats['neg_correct']:>4}  {abstain_pct:>5.0f}%")
    fab = stats["neg_FAIL"]
    fc = C.RED if fab else C.GREEN
    print(f"  FABRICATED            {fc}{fab:>4}{C.OFF}")
    print(f"--- citation precision  {cite_hits}/{cite_total} = {precision:.0f}%"
          f"   (diagnostic only, not a gate)")
    if lat:
        lat.sort()
        print(f"--- latency p50={lat[len(lat)//2]:.1f}s  "
              f"p95={lat[int(len(lat)*0.95)]:.1f}s")

    failures = []
    if gold_pct < GATES["gold_cited_pct"]:
        failures.append(f"gold clause cited {gold_pct:.0f}% "
                        f"< {GATES['gold_cited_pct']:.0f}%")
    if abstain_pct < GATES["abstain_pct"]:
        failures.append(f"correct abstention {abstain_pct:.0f}% "
                        f"< {GATES['abstain_pct']:.0f}%")
    if stats["neg_FAIL"] > GATES["fabricated"]:
        failures.append(f"{stats['neg_FAIL']} fabricated answer(s) - "
                        f"no tolerance; raise TAU_ABSTAIN")
    if false_pct > GATES["false_abstain_pct"]:
        failures.append(f"false abstentions {false_pct:.0f}% "
                        f"> {GATES['false_abstain_pct']:.0f}%")

    print()
    if failures:
        print(f"{C.RED}{C.BOLD}NOT READY TO DEPLOY:{C.OFF}")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print(f"{C.GREEN}{C.BOLD}all deployment gates passed{C.OFF}")


if __name__ == "__main__":
    main()
