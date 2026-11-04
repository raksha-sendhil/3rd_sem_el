"""Remove headers and footers that repeat on every page, keeping the first copy.

A 4-page discharge summary prints the letterhead and the patient banner on every page. Kept
four times, they flood retrieval and the agents with the same identifiers. This module is
deliberately conservative, so it never removes content:

* only the first ``TOP`` and last ``BOTTOM`` non-empty lines of a page are candidates;
* a line counts as repeated only if it is on at least half the pages (and at least 2);
* two lines are "the same" when they match after case, spacing and Markdown marks are ignored,
  or when they are long, nearly identical (OCR noise) AND carry exactly the same digits, so
  "Date: 12/03/2024" and "Date: 13/03/2024" are never merged. Only page numbers
  ("Page 2 of 4", "- 3 -") match with different digits;
* the first page keeps everything, and a page is never emptied. Every removed line is a copy
  of a line kept on an earlier page (or a page number), so no information is lost.
"""

from __future__ import annotations

import math
import re
from difflib import SequenceMatcher

from sanitise.ingest.models import RemovedLine

TOP = 12  # a letterhead + patient banner, one label per line, is about 10 lines
BOTTOM = 4
MIN_SHARE = 0.5
FUZZY_MIN_LEN = 25
FUZZY_RATIO = 0.92

_PAGE_NO = re.compile(r"^(?:page\s*)?[-\u2013(]?\s*\d+\s*(?:(?:of|/)\s*\d+)?\s*[-\u2013)]?$", re.I)
_TABLE_RULE = re.compile(r"^\|?\s*:?-{3,}")


def _clean(line: str) -> str:
    s = re.sub(r"[|*#_`>]+", " ", line.lower())
    return re.sub(r"\s+", " ", s).strip()


def _key(line: str) -> str | None:
    """The comparison key of a line, or None if it is not a header/footer candidate."""
    s = _clean(line)
    if not s or _TABLE_RULE.match(line.strip()):
        return None
    if _PAGE_NO.match(s):
        return "<page-number>"
    if sum(c.isalpha() for c in s) < 3:
        return None
    return s


def _digits(s: str) -> str:
    return "".join(c for c in s if c.isdigit())


def _same(a: str, b: str) -> bool:
    if a == b:
        return True
    if min(len(a), len(b)) < FUZZY_MIN_LEN or _digits(a) != _digits(b):
        return False
    return SequenceMatcher(None, a, b).ratio() >= FUZZY_RATIO


def _zone(lines: list[str]) -> list[int]:
    """Indexes of the first TOP and last BOTTOM non-empty lines."""
    filled = [i for i, ln in enumerate(lines) if ln.strip()]
    return sorted(set(filled[:TOP]) | set(filled[-BOTTOM:]))


def strip_repeated(pages: list[str]) -> tuple[list[str], list[RemovedLine]]:
    """``pages`` with repeated header/footer lines removed after their first page."""
    n = len(pages)
    if n < 2:
        return list(pages), []
    split = [p.split("\n") for p in pages]
    # clusters of matching keys: representative key -> set of pages it is on
    reps: list[str] = []
    on_pages: list[set[int]] = []
    found: list[list[tuple[int, int]]] = []  # per page: (line index, cluster)
    for p, lines in enumerate(split):
        here: list[tuple[int, int]] = []
        for i in _zone(lines):
            key = _key(lines[i])
            if key is None:
                continue
            c = next((j for j, rep in enumerate(reps) if _same(key, rep)), None)
            if c is None:
                reps.append(key)
                on_pages.append(set())
                c = len(reps) - 1
            on_pages[c].add(p)
            here.append((i, c))
        found.append(here)

    needed = max(2, math.ceil(MIN_SHARE * n))
    repeated = {c for c, ps in enumerate(on_pages) if len(ps) >= needed}
    first_page = {c: min(on_pages[c]) for c in repeated}

    out: list[str] = []
    removed: list[RemovedLine] = []
    for p, lines in enumerate(split):
        drop = {i for i, c in found[p] if c in repeated and p > first_page[c]}
        filled = sum(1 for ln in lines if ln.strip())
        if drop and len(drop) >= filled:  # would empty the page: leave it alone
            drop = set()
        removed += [RemovedLine(page=p, line=lines[i].strip()) for i in sorted(drop)]
        kept = [ln for i, ln in enumerate(lines) if i not in drop]
        out.append(re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip())
    return out, removed
