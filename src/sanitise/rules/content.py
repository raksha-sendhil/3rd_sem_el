"""Deterministic gate after Agent 1 (legal): what is content and identifies nobody.

A false detection is destructive, the span is replaced in every fact that contains it ("the
Supplier shall" became "a private company shall", "30 days' notice" lost its 30 days).

* DATE without a calendar anchor: a duration or count. A date of birth is never dropped.
* The defined role words of a contract or matter ("the Supplier", "Licensee", "the Territory"):
  they name a part in the agreement, never who plays it.
"""

from __future__ import annotations

import re

from sanitise.schemas import SensitiveSpan

_CALENDAR = re.compile(
    r"\b(?:jan|feb|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b"
    r"|\b(?:january|february|april|june|july|august|september|october|november|december)\b"
    r"|\b(?:mon|tues|wednes|thurs|fri|satur|sun)day\b"
    r"|\b(?:christmas|new year|easter|diwali|deepavali|holi|eid|ramadan|ramzan|pongal|onam"
    r"|navratri|dussehra|hanukkah|thanksgiving|halloween|independence day|republic day)\b"
    r"|(?<!\d)(?:19|20)\d{2}(?!\d)(?!\s*(?:mg|mcg|ml|iu|units?|calories|kcal|steps)\b)"
    r"|(?<!\d)\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}(?!\d)|(?<![\d.])\d{1,2}/\d{2,4}(?!\d)"
    r"|(?<!\d)\d{8}(?!\d)|(?<!\w)'\d{2}(?!\d)"
    r"|\b\d{1,2}(?:st|nd|rd|th)\s+(?:of\s+)?(?:this|last|next)\s+month\b",
    re.IGNORECASE,
)
# "May" and "March" are also words: only with a capital, or next to a day number
_MAY_MARCH = re.compile(
    r"\b(?:May|March|Mar|MAY|MARCH|MAR)\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:may|march|mar)\b"
    r"|\b(?:may|march|mar)\s+\d{1,2}\b"
)
_BIRTH = re.compile(r"(?:\bborn\b|\bd\.?o\.?b\b|\bbirth)[^.\n]{0,20}$", re.IGNORECASE)
_ROLE = re.compile(
    r"(?:the\s+)?(?:company|corporation|supplier|vendor|purchaser|buyer|seller|customer|client"
    r"|licen[sc]or|licen[sc]ee|distributor|manufacturer|contractor|sub-?contractor|consultant"
    r"|franchisor|franchisee|lessor|lessee|landlord|tenant|borrower|lender|guarantor|employer"
    r"|employee|agent|principal|reseller|developer|operator|owner|provider|service provider"
    r"|recipient|sponsor|investor|partner|parties|party|affiliates?|subsidiary|agreement"
    r"|contract|territory|products?|services?|petitioner|respondent|plaintiff|defendant"
    r"|appellant|complainant|accused)(?:s|['\u2019]s|['\u2019])?",
    re.IGNORECASE,
)
_DEFINED = re.compile(r"(?:\bthe\s+|[\"\u201c(])$", re.IGNORECASE)
_NAMING = {"ORGANIZATION", "NAME", "OTHER", "LOCATION", "MATTER_CODENAME"}


def not_identifying(text: str, span: SensitiveSpan) -> bool:
    """True when the span is content that identifies nobody (see the module docstring)."""
    if span.start is None or span.end is None:
        return False
    value = " ".join(span.text.split()).strip(" ,;:.!?\"'()\u201c\u201d")
    before = text[max(0, span.start - 60) : span.start]
    kind = span.category.value
    if kind == "DATE":
        dated = _CALENDAR.search(span.text) or _MAY_MARCH.search(span.text)
        return not dated and not _BIRTH.search(before)
    if not _ROLE.fullmatch(value):
        return False
    if kind in _NAMING:
        return True
    # a job can be a role word ("works as a contractor"): only the defined term is content
    return kind == "OCCUPATION" and value[:1].isupper() and bool(_DEFINED.search(before))
