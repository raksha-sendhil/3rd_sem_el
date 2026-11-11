"""Agent 3 - leak check and fallback masking.  OWNER: Person A (Deterministic Logic).

* ``leak_check(removed, safe_text)``: ``removed`` are the decisions whose action is
  placeholder or generalise. Returns one ``SensitiveSpan`` (offsets in ``safe_text``,
  ``method="rule"``) for every place their original text still appears, including close
  variants (legal build spec, step 8):

  - **spacing and punctuation**: ``Rohan  Mehta``, ``rohan-mehta``, ``O.S. No 412 / 2024``
  - **digits with other separators** for numbers (phones, IDs, accounts, case numbers,
    amounts): ``9845012345`` for ``+91 98450 12345``, ``ABCPM 1234 K`` for ``ABCPM1234K``
  - **name parts**: ``Mehta`` alone when ``Rohan Mehta`` was removed; also for companies,
    addresses and matter code names, skipping generic words (``Builders``, ``Park``)
  - **near spellings** (rapidfuzz, ratio >= 90) for names, companies, addresses, code names:
    ``Rohan Mheta``

  A word that is part of any decision's own replacement ("a city in Maharashtra") is never
  reported as a leak.

* ``mask_fallback(safe_text, leaks, placeholders)``: replaces every leaked span
  deterministically with what its decision replaced it with (the placeholder token or the
  generalised phrase, carried in the leak's rationale after ``->``), otherwise ``[REDACTED]``.
  The pipeline uses it when a rewrite still leaks after one retry.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from sanitise_legal.schemas import Action, PIICategory, RoutingDecision, SensitiveSpan

C = PIICategory

_NUMERIC = {C.PHONE, C.ID_NUMBER, C.FINANCIAL, C.CASE_NUMBER, C.MONEY, C.ADDRESS}
_WORDY = {C.NAME, C.ORGANIZATION, C.ADDRESS, C.MATTER_CODENAME, C.EMAIL}
_MIN_DIGITS = 5  # shorter digit runs ("2024", "138") are too common to call a leak
_FUZZY_MIN_LEN = 6
_FUZZY_RATIO = 90
# Words that are part of names of things but identify nobody on their own.
_GENERIC_WORDS = {
    "the",
    "and",
    "of",
    "for",
    "pvt",
    "ltd",
    "private",
    "limited",
    "llp",
    "inc",
    "corp",
    "corporation",
    "co",
    "company",
    "sons",
    "associates",
    "enterprises",
    "industries",
    "traders",
    "builders",
    "developers",
    "construction",
    "constructions",
    "infra",
    "realty",
    "bank",
    "finance",
    "motors",
    "textiles",
    "exports",
    "pharma",
    "logistics",
    "solutions",
    "technologies",
    "group",
    "project",
    "operation",
    "flat",
    "plot",
    "house",
    "shop",
    "office",
    "unit",
    "road",
    "rd",
    "street",
    "lane",
    "marg",
    "nagar",
    "colony",
    "society",
    "park",
    "layout",
    "main",
    "cross",
    "apartment",
    "apartments",
    "tower",
    "towers",
    "block",
    "wing",
    "floor",
    "near",
    "opp",
    "east",
    "west",
    "north",
    "south",
    "new",
    "old",
    "mr",
    "mrs",
    "ms",
    "dr",
    "shri",
    "smt",
    "adv",
}


def _rationale(kind: str, decision: RoutingDecision) -> str:
    return (
        f"leak check: {kind} of removed {decision.category.value} -> {decision.replacement or ''}"
    )


def _span(safe_text: str, start: int, end: int, kind: str, d: RoutingDecision) -> SensitiveSpan:
    return SensitiveSpan(
        text=safe_text[start:end],
        category=d.category,
        start=start,
        end=end,
        method="rule",
        rationale=_rationale(kind, d),
    )


def _loose_pattern(original: str) -> re.Pattern | None:
    """The original's alphanumeric runs, with any spacing or punctuation between them."""
    runs = re.findall(r"[0-9A-Za-z]+", original)
    if not runs:
        return None
    return re.compile(
        r"(?<![0-9A-Za-z])" + r"[\W_]*".join(map(re.escape, runs)) + r"(?![0-9A-Za-z])", re.I
    )


def _digit_runs(safe_text: str, digits: str, prefix: str = "") -> list[tuple[int, int]]:
    """Places where ``digits`` appear in order, separated only by spaces and punctuation."""
    pattern = r"(?<!\d)" + prefix + r"[\s().+/-]*".join(digits) + r"(?!\d)"
    return [m.span() for m in re.finditer(pattern, safe_text)]


def _alnum_runs(safe_text: str, chars: str) -> list[tuple[int, int]]:
    pattern = r"(?<![0-9A-Za-z])" + r"[\s.-]*".join(map(re.escape, chars)) + r"(?![0-9A-Za-z])"
    return [m.span() for m in re.finditer(pattern, safe_text, re.I)]


def _protected_words(decisions: list[RoutingDecision]) -> set[str]:
    words: set[str] = set()
    for d in decisions:
        words.update(w.lower() for w in re.findall(r"[A-Za-z]+", d.replacement or ""))
    return words


def leak_check(removed: list[RoutingDecision], safe_text: str) -> list[SensitiveSpan]:
    candidates: list[SensitiveSpan] = []
    protected = _protected_words(removed)
    tokens = [(m.start(), m.end()) for m in re.finditer(r"\S+", safe_text)]

    def add(start: int, end: int, kind: str, d: RoutingDecision) -> None:
        candidates.append(_span(safe_text, start, end, kind, d))

    for d in removed:
        original = d.span_text.strip()
        if d.action is Action.KEEP or not original:
            continue

        # 1. the original, ignoring case, spacing and punctuation
        if (pattern := _loose_pattern(original)) is not None:
            for m in pattern.finditer(safe_text):
                add(m.start(), m.end(), "exact or spacing variant", d)

        # 2. numbers written with other separators (+91 98450 12345 -> 9845012345)
        if d.category in _NUMERIC:
            digits, prefix = re.sub(r"\D", "", original), ""
            if d.category is C.PHONE and len(digits) >= 10:
                # the last 10 digits, with or without a +91 or 0 in front
                digits, prefix = digits[-10:], r"(?:(?:\+?91|0)[\s().-]*)?"
            if len(digits) >= _MIN_DIGITS:
                for start, end in _digit_runs(safe_text, digits, prefix):
                    add(start, end, "digits", d)
            chars = re.sub(r"[^0-9A-Za-z]", "", original)
            if len(chars) >= 8 and chars != digits and any(c.isdigit() for c in chars):
                for start, end in _alnum_runs(safe_text, chars):  # ABCPM 1234 K
                    add(start, end, "separator variant", d)

        if d.category not in _WORDY:
            continue

        # 3. a distinctive part on its own: "Mehta" from "Rohan Mehta"
        for word in re.findall(r"[A-Za-z][A-Za-z'’]+", original):
            if len(word) < 3 or word.lower() in _GENERIC_WORDS or word.lower() in protected:
                continue
            if d.category is not C.NAME and not word[0].isupper():
                continue
            for m in re.finditer(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", safe_text, re.I):
                add(m.start(), m.end(), "name part", d)

        # 4. near spellings over windows of the same number of words
        if len(original) >= _FUZZY_MIN_LEN:
            n = max(1, len(original.split()))
            target = original.lower()
            for i in range(len(tokens) - n + 1):
                start, end = tokens[i][0], tokens[i + n - 1][1]
                window = safe_text[start:end].strip(".,;:!?()\"'")
                if "[" in window or not window:
                    continue
                if fuzz.ratio(window.lower(), target) >= _FUZZY_RATIO:
                    w_start = safe_text.index(window, start)
                    add(w_start, w_start + len(window), "near spelling", d)

    # longest first, so "Rohan Mheta" is one leak rather than "Rohan" plus a missed "Mheta"
    leaks: list[SensitiveSpan] = []
    for c in sorted(candidates, key=lambda s: (s.start or 0) - (s.end or 0)):
        if not any(c.start < k.end and k.start < c.end for k in leaks):
            leaks.append(c)
    return sorted(leaks, key=lambda s: s.start or 0)


_REPLACEMENT = re.compile(r" -> (.+)$")


def mask_fallback(safe_text: str, leaks: list[SensitiveSpan], placeholders: dict[str, str]) -> str:
    token_for = {original.lower(): token for token, original in placeholders.items()}
    out = safe_text
    for leak in sorted(leaks, key=lambda s: s.start or 0, reverse=True):
        assert leak.start is not None and leak.end is not None
        match = _REPLACEMENT.search(leak.rationale or "")
        replacement = (match.group(1).strip() if match else "") or token_for.get(
            leak.text.lower(), "[REDACTED]"
        )
        out = out[: leak.start] + replacement + out[leak.end :]
    return out
