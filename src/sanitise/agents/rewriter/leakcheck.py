"""Agent 3 - leak check and fallback masking.  OWNER: Person A (Deterministic Logic).

``leak_check(removed, safe_text)`` looks for every detail that was supposed to be removed
(decisions whose action is placeholder or generalise) and returns one ``SensitiveSpan`` per
place it still appears, with offsets in ``safe_text`` and ``method="rule"``. A detail counts
as present when ``safe_text`` contains it in any of these forms:

* the same words, ignoring case, spacing and punctuation between them
  (``6 ft 3 inches`` / ``6ft 3 inches``, ``Fort Kochi`` / ``fort-kochi``);
* NAME: any single part of the name on its own, as a capitalised word ("Laura" when
  "Laura Fenn" was removed), except titles and parts shorter than three letters;
* PHONE: the last ten digits (or all, if fewer) with any separators (``98450-12345``);
* ID_NUMBER: the same letters and digits with any separators, or its digits alone if there
  are at least five (``HC 2291746``, ``2291746``);
* EMAIL: the part before the ``@`` on its own (``rajesh.kumar84``);
* ADDRESS and LOCATION: any comma-separated part that is distinctive on its own (one with a
  digit or two words, for an address; any part of four letters or more, for a location);
* AGE: the number with an age word (``32-year-old``, ``aged 32``);
* DATE: the day and month in either order or as a numeric date (``March 3``, ``3/3/2022``);
* PHYSICAL: each number with its unit (``220 lbs`` for ``approximately 220 pounds``).

Text inside placeholder tokens (``[PERSON_1]``, ``[REDACTED]``) is never reported. Overlapping
matches are reported once, as the longest.

``mask_fallback(safe_text, leaks, placeholders)`` replaces every leak deterministically: with
the placeholder token whose original it belongs to (a first name gets the full name's token,
a reformatted phone number its phone token), otherwise with ``[REDACTED]``. The pipeline uses
it when a rewrite still leaks after one retry; the masked text passes ``leak_check``.

``leak_patterns(text, category)`` is the same matcher for one detail; ``ontology`` uses it to
make sure a generalised phrase never repeats what it replaces.
"""

from __future__ import annotations

import re
from functools import lru_cache

from sanitise.schemas import Action, PIICategory, RoutingDecision, SensitiveSpan

C = PIICategory

REDACTED = "[REDACTED]"
_TOKEN = re.compile(r"\[[A-Z]+(?:_[A-Z]+)*(?:_\d+)?\]")
_WORD = re.compile(r"[A-Za-z0-9]+")
_SEP = r"[\s\-./()+]*"
_NAME_SKIP = {
    "mr", "mrs", "ms", "miss", "dr", "prof", "sir", "madam", "jr", "sr", "van", "von", "de",
    "da", "del", "la", "le", "al", "bin", "binti", "the", "and",
}  # fmt: skip
_MONTHS = [
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december",
]  # fmt: skip
_AGE_UNIT = r"(?:years?|yrs?|y/?o|year[\s-]old|years[\s-]old)"
_WEIGHT_UNIT = r"(?:lbs?|pounds?|kgs?|kilograms?|kilos?|stone)"
_LENGTH_UNIT = r"(?:cm|cms|centimet(?:er|re)s?|m|meters?|metres?)"


def _bounded(body: str) -> str:
    return r"(?<![A-Za-z0-9])" + body + r"(?![A-Za-z0-9])"


def _flex(text: str) -> str | None:
    """The words of ``text`` in order, with up to three non-word characters between them."""
    words = _WORD.findall(text)
    return _bounded(r"[\W_]{0,3}".join(map(re.escape, words))) if words else None


def _month_alt(month: int) -> str:
    name = _MONTHS[month - 1]
    return f"(?:{name}|{name[:3]}\\.?{'|sept' if month == 9 else ''})"


def _date_variants(text: str) -> list[str]:
    lower = text.lower()
    month = next(
        (i + 1 for i, name in enumerate(_MONTHS) if re.search(rf"\b{name[:3]}", lower)), None
    )
    days = [int(d) for d in re.findall(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", lower) if 0 < int(d) < 32]
    years = re.findall(r"\b((?:19|20)\d{2})\b", lower)
    out = []
    if month and days:
        day, m = days[0], _month_alt(month)
        out += [rf"\b{day}(?:st|nd|rd|th)?\s+(?:of\s+)?{m}\b", rf"\b{m}\s+{day}(?:st|nd|rd|th)?\b"]
        for y in years:
            yy = f"(?:{y}|{y[2:]})"
            out += [rf"\b0?{day}[/.-]0?{month}[/.-]{yy}\b", rf"\b0?{month}[/.-]0?{day}[/.-]{yy}\b"]
    return out


def _physical_variants(text: str) -> list[str]:
    out = []
    feet = re.search(r"(\d)\s*(?:ft|feet|foot|'|\u2019)\.?\s*(?:and\s+)?(\d{1,2})?", text, re.I)
    if feet:
        inches = rf"\s*(?:and\s+)?{feet.group(2)}\b" if feet.group(2) else ""
        out.append(rf"\b{feet.group(1)}\s*(?:ft|feet|foot|'|\u2019)\.?{inches}")
    for n in re.findall(rf"(\d+(?:\.\d+)?)\s*{_WEIGHT_UNIT}\b", text, re.I):
        out.append(rf"\b{re.escape(n)}\s*{_WEIGHT_UNIT}(?![a-z])")
    for n in re.findall(rf"(\d+(?:\.\d+)?)\s*{_LENGTH_UNIT}\b", text, re.I):
        out.append(rf"\b{re.escape(n)}\s*{_LENGTH_UNIT}\b")
    return out


@lru_cache(maxsize=4096)
def _patterns(text: str, category: PIICategory) -> tuple[tuple[str, re.Pattern[str]], ...]:
    found: list[tuple[str, str, int]] = []  # (kind, regex, flags)
    whole = _flex(text)
    words = _WORD.findall(text)
    # a lone short number ("28", an age) only counts with an age word next to it
    if whole and not (category is C.AGE and len(words) == 1):
        found.append(("same words", whole, re.IGNORECASE))
    if category is C.NAME:
        for part in re.findall(r"[A-Za-z][A-Za-z'\u2019-]*", text):
            if len(part) >= 3 and part.lower() not in _NAME_SKIP and len(words) > 1:
                found.append(
                    ("part of the name", _bounded(re.escape(part[0].upper() + part[1:])), 0)
                )
    elif category is C.PHONE:
        digits = re.sub(r"\D", "", text)
        if len(digits) >= 7:
            found.append(("digits", _bounded(_SEP.join(digits[-10:])), 0))
    elif category is C.ID_NUMBER:
        chars = re.sub(r"[^A-Za-z0-9]", "", text)
        if len(chars) >= 4:
            found.append(("reformatted", _bounded(_SEP.join(map(re.escape, chars))), re.I))
        digits = re.sub(r"\D", "", text)
        if len(digits) >= 5 and digits != chars:
            found.append(("digits", r"(?<!\d)" + digits + r"(?!\d)", 0))
    elif category is C.EMAIL and "@" in text:
        local = text.split("@", 1)[0]
        if len(local) >= 4 and (pattern := _flex(local)):
            found.append(("user name", pattern, re.IGNORECASE))
    elif category in (C.ADDRESS, C.LOCATION):
        for part in (p.strip() for p in text.split(",")):
            distinctive = (
                bool(re.search(r"\d", part)) or len(_WORD.findall(part)) >= 2
                if category is C.ADDRESS
                else len(re.sub(r"[^A-Za-z]", "", part)) >= 4
            )
            if part != text.strip() and distinctive and (pattern := _flex(part)):
                found.append(("part of the place", pattern, re.IGNORECASE))
            # house number and street name, whatever the suffix: "1400 Smith St"
            street = re.match(r"(\d+[A-Za-z]?(?:/\d+)?)\s+([A-Za-z][A-Za-z'\u2019-]{2,})", part)
            if category is C.ADDRESS and street:
                found.append(("street", _flex(f"{street[1]} {street[2]}") or "", re.IGNORECASE))
    elif category is C.AGE:
        for n in re.findall(r"\d{1,3}", text)[:1]:
            found.append(("age", rf"\b{n}[\s-]*{_AGE_UNIT}\b", re.IGNORECASE))
            found.append(("age", rf"\b(?:aged?|age\s+of)\s*:?\s*{n}\b", re.IGNORECASE))
    elif category is C.DATE:
        found += [("date", p, re.IGNORECASE) for p in _date_variants(text)]
    elif category is C.PHYSICAL:
        found += [("measurement", p, re.IGNORECASE) for p in _physical_variants(text)]
    seen: set[str] = set()
    out = []
    for kind, regex, flags in found:
        if regex not in seen:
            seen.add(regex)
            out.append((kind, re.compile(regex, flags)))
    return tuple(out)


def leak_patterns(text: str, category: PIICategory) -> list[re.Pattern[str]]:
    return [p for _, p in _patterns(" ".join(text.split()), category)]


def contains(haystack: str, text: str, category: PIICategory) -> bool:
    """Whether ``haystack`` still contains ``text`` in any form the leak check looks for."""
    return any(p.search(haystack) for p in leak_patterns(text, category))


def leak_check(removed: list[RoutingDecision], safe_text: str) -> list[SensitiveSpan]:
    protected = [(m.start(), m.end()) for m in _TOKEN.finditer(safe_text)]
    hits: list[tuple[int, int, PIICategory, str]] = []
    for decision in removed:
        if decision.action is Action.KEEP or not decision.span_text.strip():
            continue
        for kind, pattern in _patterns(" ".join(decision.span_text.split()), decision.category):
            for m in pattern.finditer(safe_text):
                if m.start() == m.end() or any(a < m.end() and m.start() < b for a, b in protected):
                    continue
                hits.append((m.start(), m.end(), decision.category, kind))
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    leaks: list[SensitiveSpan] = []
    for start, end, category, kind in hits:
        if leaks and start < (leaks[-1].end or 0):
            continue  # overlaps a longer match already reported
        leaks.append(
            SensitiveSpan(
                text=safe_text[start:end],
                category=category,
                start=start,
                end=end,
                method="rule",
                rationale=f"leak check ({kind}): removed {category.value} still present",
            )
        )
    return leaks


def _token_for(leak: SensitiveSpan, placeholders: dict[str, str]) -> str:
    for token, original in placeholders.items():
        if contains(original, leak.text, leak.category) or any(
            p.fullmatch(leak.text) for p in leak_patterns(original, leak.category)
        ):
            return token
    return REDACTED


def mask_fallback(safe_text: str, leaks: list[SensitiveSpan], placeholders: dict[str, str]) -> str:
    out, last_start = safe_text, len(safe_text) + 1
    for leak in sorted(leaks, key=lambda s: s.start or 0, reverse=True):
        assert leak.start is not None and leak.end is not None
        if leak.end > last_start:  # overlaps a leak already masked
            continue
        out = out[: leak.start] + _token_for(leak, placeholders) + out[leak.end :]
        last_start = leak.start
    return out
