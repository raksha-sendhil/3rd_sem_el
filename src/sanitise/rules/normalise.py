"""Span normalisation: trim, deduplicate, resolve category conflicts, merge neighbours.

All functions are pure and deterministic. Apart from the offset check and the
adjacent-merge step, they work from each span's own ``text`` and offsets, so they
can also be used by ``merge_spans`` which never sees the document.

Invariant on output: for every positional span, ``text == doc[start:end]``, and no
two positional spans overlap.
"""

import re

from sanitise.rules.categories import precedence
from sanitise.schemas import PIICategory, SensitiveSpan

C = PIICategory


class OffsetOutOfRangeError(ValueError):
    """A span's offsets fall outside the document or do not match its text.

    Maps to the OFFSET_OUT_OF_RANGE code in the error envelope.
    """


_LEAD_TRIM = " \t\r\n,;:\"'`“”‘’"
_TRAIL_TRIM = " \t\r\n.,;:!?\"'`“”‘’"
_UNIT_MARKS = "\"'”’"
_POSSESSIVE = re.compile(r"(?:'s|’s|'|’)$")
_POSSESSIVE_CATEGORIES = {C.NAME, C.LOCATION}

# Residual fragments left after a lower-precedence span loses an overlap are kept
# (conservative: recall first) unless they are trivially uninformative.
_RESIDUAL_STOPWORDS = {
    "a",
    "an",
    "the",
    "my",
    "his",
    "her",
    "our",
    "their",
    "of",
    "and",
    "or",
    "in",
    "on",
    "at",
    "to",
    "is",
    "was",
    "be",
    "as",
    "by",
    "for",
    "with",
    "old",
    "year",
    "years",
}

# What may sit between two same-category spans for them to count as adjacent.
_DEFAULT_GAP = re.compile(r"\s{0,2}")
_GAPS = {
    C.ADDRESS: re.compile(r"[\s,.]{0,3}"),  # "1400 Smith St., Houston, TX"
    C.LOCATION: re.compile(r"[\s,]{0,3}"),
    C.DATE: re.compile(r"[\s,]{0,3}"),
}


def is_positional(span: SensitiveSpan) -> bool:
    return span.start is not None and span.end is not None


def check_offsets(spans: list[SensitiveSpan], text: str) -> None:
    for s in spans:
        if not is_positional(s):
            continue
        if not (0 <= s.start < s.end <= len(text)):
            raise OffsetOutOfRangeError(
                f"span {s.text!r} [{s.start}:{s.end}] outside document of length {len(text)}"
            )
        if text[s.start : s.end] != s.text:
            raise OffsetOutOfRangeError(
                f"span text {s.text!r} != document[{s.start}:{s.end}] {text[s.start : s.end]!r}"
            )


def _slice(span: SensitiveSpan, start: int, end: int) -> SensitiveSpan:
    """Sub-span of a positional span, cut from its own text."""
    return span.model_copy(
        update={
            "text": span.text[start - span.start : end - span.start],
            "start": start,
            "end": end,
        }
    )


def trim(span: SensitiveSpan) -> SensitiveSpan | None:
    """Strip surrounding whitespace, punctuation and (for names/places) possessives.

    Returns None if nothing alphanumeric remains.
    """
    if not is_positional(span):
        return span
    t = span.text
    lo, hi = 0, len(t)
    while lo < hi and t[lo] in _LEAD_TRIM:
        lo += 1
    while hi > lo and t[hi - 1] in _TRAIL_TRIM:
        if t[hi - 1] in _UNIT_MARKS and hi - 2 >= lo and t[hi - 2].isdigit():
            break  # 5'4" - feet/inch marks, not quotation
        hi -= 1
    if span.category in _POSSESSIVE_CATEGORIES:
        m = _POSSESSIVE.search(t, lo, hi)
        if m and m.start() > lo:
            hi = m.start()
    if not any(ch.isalnum() for ch in t[lo:hi]):
        return None
    if (lo, hi) == (0, len(t)):
        return span
    return _slice(span, span.start + lo, span.start + hi)


def _join_rationale(a: str | None, b: str | None) -> str | None:
    parts = []
    for r in (a, b):
        for p in (r or "").split("; "):
            if p and p not in parts:
                parts.append(p)
    return "; ".join(parts) or None


def _join(a: SensitiveSpan, b: SensitiveSpan, gap: str = "") -> SensitiveSpan:
    """Combine two same-category spans with a.start <= b.start that overlap or touch
    (``gap`` is the document text between them when they don't)."""
    if b.end <= a.end:
        text, end = a.text, a.end
    elif b.start <= a.end:
        text, end = a.text + b.text[a.end - b.start :], b.end
    else:
        text, end = a.text + gap + b.text, b.end
    return a.model_copy(
        update={
            "text": text,
            "end": end,
            "method": "rule" if "rule" in (a.method, b.method) else "llm",
            "confidence": max(a.confidence, b.confidence),
            "rationale": _join_rationale(a.rationale, b.rationale),
        }
    )


def union_same_category(spans: list[SensitiveSpan]) -> list[SensitiveSpan]:
    """Collapse overlapping spans that share a category into one covering span."""
    out: list[SensitiveSpan] = []
    by_cat: dict[PIICategory, list[SensitiveSpan]] = {}
    for s in spans:
        by_cat.setdefault(s.category, []).append(s)
    for group in by_cat.values():
        group.sort(key=lambda s: (s.start, -s.end))
        cur = group[0]
        for s in group[1:]:
            if s.start < cur.end:
                cur = _join(cur, s)
            else:
                out.append(cur)
                cur = s
        out.append(cur)
    return out


def _meaningful(span: SensitiveSpan) -> bool:
    alnum = sum(ch.isalnum() for ch in span.text)
    return alnum >= 2 and span.text.lower() not in _RESIDUAL_STOPWORDS


def resolve_overlaps(spans: list[SensitiveSpan]) -> list[SensitiveSpan]:
    """Resolve overlaps between different categories by fixed precedence.

    Spans are accepted greedily, strongest first (precedence, then longer, then
    more confident, then rule before llm). A weaker span that overlaps accepted
    ones is cut down to the parts nobody claimed; those residual pieces survive
    only if they still carry content, so partial overlaps never lose coverage.
    """
    order = sorted(
        spans,
        key=lambda s: (
            precedence(s.category),
            -(s.end - s.start),
            -s.confidence,
            s.method != "rule",
            s.start,
        ),
    )
    accepted: list[SensitiveSpan] = []
    for cand in order:
        pieces = [cand]
        for acc in accepted:
            nxt = []
            for p in pieces:
                if p.end <= acc.start or acc.end <= p.start:
                    nxt.append(p)
                    continue
                if p.start < acc.start:
                    nxt.append(_slice(p, p.start, acc.start))
                if acc.end < p.end:
                    nxt.append(_slice(p, acc.end, p.end))
            pieces = nxt
        for p in pieces:
            if p is cand:
                accepted.append(p)
            elif (t := trim(p)) is not None and _meaningful(t):
                accepted.append(t)
    return accepted


def merge_adjacent(spans: list[SensitiveSpan], text: str) -> list[SensitiveSpan]:
    """Join consecutive same-category spans separated only by whitespace
    (or commas, for addresses, locations and dates)."""
    spans = sorted(spans, key=lambda s: s.start)
    out: list[SensitiveSpan] = []
    for s in spans:
        if out and out[-1].category == s.category:
            prev = out[-1]
            gap = text[prev.end : s.start]
            if _GAPS.get(s.category, _DEFAULT_GAP).fullmatch(gap):
                out[-1] = _join(prev, s, gap)
                continue
        out.append(s)
    return out


def normalise_spans(spans: list[SensitiveSpan], text: str | None = None) -> list[SensitiveSpan]:
    """Trim -> union same-category overlaps -> resolve cross-category overlaps ->
    merge adjacent same-category spans. Non-positional spans pass through untouched.

    ``text`` enables offset validation and the adjacent merge; without it only the
    text-free steps run.
    """
    positional = [s for s in spans if is_positional(s)]
    rest = [s for s in spans if not is_positional(s)]
    if text is not None:
        check_offsets(positional, text)
    positional = [t for s in positional if (t := trim(s)) is not None]
    if positional:
        positional = resolve_overlaps(union_same_category(positional))
        if text is not None:
            positional = merge_adjacent(positional, text)
    positional.sort(key=lambda s: (s.start, s.end))
    return positional + rest
