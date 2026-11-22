"""Deterministic gate after Agent 1: what is content and identifies nobody.  OWNER: Person A.

spaCy, the recognisers and the LLM pass all over-detect, and a false detection is destructive: the
span is replaced in every fact that contains it ("the doctor advised" became "a working
professional advised", "2D Echo" became "an earlier date Echo"). ``not_identifying`` drops such
a span, by rules only:

* DATE without a calendar anchor (month, weekday, year, numeric date): a duration or count
  ("3 months", "2D", "5 minutes"). A date of birth is never dropped.
* OCCUPATION that is the role of a speaker or of the service ("Customer", "Agent", "customer
  service representative"), unless the text says it is someone's job ("I am a nurse").
* LOCATION made only of common facility nouns in lower case ("nursing facility", "school").
* PHYSICAL that is a sex or the size of a finding ("female", "7 inches"), not a body measure.
* a "URL" made by a missing space after a full stop ("side.Co", "area.pl" of "area.please").

Everything else is left as detected: when unsure, a span stays sensitive.
"""

from __future__ import annotations

import re

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
    r"(?:(?:the|a|an|bank|call|phone|support|customer|service|care|chat|claims?|credit|card"
    r"|loan|account|collection|collections)\s+){0,4}"
    r"(?:customers?|agents?|representatives?|reps?|advisors?|callers?|cardholders?"
    r"|account\s+holders?|policy\s*holders?|borrowers?|lenders?|applicants?)\.?",
    re.IGNORECASE,
)
_OWN_JOB = re.compile(
    r"(?:(?:\bi\s+am|\bi['’]?m)(?!\s+(?:seeing|visiting|consulting|going|waiting|under|referred))"
    r"|\bwork(?:s|ing|ed)?\s+as|\bemployed\s+as|\b(?:being|as)\s+an?|\bprofession\s*:"
    r"|\boccupation\s*:)"
    r"\s*(?:an?\s+)?"
    r"(?:(?:\d{1,3}|years?|yrs?|yr|y/o|yo|old|male|female|man|woman|\d+-year-old),?\s+){0,6}"
    r"(?:[\w-]+\s+){0,2}$",
    re.IGNORECASE,
)
_JOB_AFTER = re.compile(
    r"\s+(?:working|by\s+profession|myself|on\s+night\s+shifts?)\b", re.IGNORECASE
)
_SOMEONES_JOB = re.compile(
    r"\b(?:wife|husband|son|daughter|father|mother|brother|sister|he|she)"
    r"(?:\s+(?:is|was|works\s+as)|['’]s)\s+(?:an?\s+)?(?:[\w-]+\s+){0,2}$",
    re.IGNORECASE,
)

_GENERIC_PLACE = {
    "a", "an", "the", "my", "nursing", "facility", "school", "hospital", "clinic", "home", "house",
    "college", "university", "office", "work", "town", "city", "village", "pharmacy", "lab",
    "laboratory", "emergency", "room", "er", "icu", "ward", "opd", "gym", "care", "centre",
    "center", "medical", "local", "nearby", "nearest", "urgent", "eye", "dental",
    "bank", "branch", "atm", "online", "store", "shop",
}  # fmt: skip
_SEX = re.compile(r"(?:an?\s+)?(?:male|female|man|woman|boy|girl|lady|gentleman)", re.IGNORECASE)
_LENGTH_ONLY = re.compile(
    r"(?:about |approx\.? |approximately )?(\d+(?:\.\d+)?)\s*(?:mm|cm|cms|inch|inches|in)\.?",
    re.IGNORECASE,
)
_HEIGHT_CUE = re.compile(r"(?:\bheight|\btall|\bi\s+am|\bi['’]m)[^.\n]{0,15}$", re.IGNORECASE)


def _service_role(value: str, before: str, after: str) -> bool:
    if not _ROLE.fullmatch(value):
        return False
    if re.fullmatch(r"(?:the\s+)?(?:customers?|callers?)", value, re.IGNORECASE):
        return True
    return not (
        _OWN_JOB.search(before) or _JOB_AFTER.match(after) or _SOMEONES_JOB.search(before)
    )


def _fused_sentence(value: str, after: str) -> bool:
    if re.match(r"(?:https?://|www\.)", value, re.IGNORECASE):
        return False
    if "." not in value or ".." in value or after[:1].isalpha():
        return True
    ending = value.split("/")[0].rsplit(".", 1)[-1]
    return not (ending.islower() or value.isupper())


def not_identifying(text: str, span) -> bool:
    """True when the span is content that identifies nobody (see the module docstring)."""
    if span.start is None or span.end is None:
        return False
    value = " ".join(span.text.split()).strip(" ,;:!?\"'")
    before = text[max(0, span.start - 60) : span.start]
    after = text[span.end : span.end + 30]
    kind = span.category.value
    if kind == "DATE":
        dated = _CALENDAR.search(span.text) or _MAY_MARCH.search(span.text)
        return not dated and not _BIRTH.search(before)
    if kind == "OCCUPATION":
        return _service_role(value, before, after)
    if kind == "PHYSICAL":
        if _SEX.fullmatch(value):
            return True
        length = _LENGTH_ONLY.fullmatch(value)
        return bool(length) and float(length[1]) < 48 and not _HEIGHT_CUE.search(before)
    if kind == "OTHER" and "UrlRecognizer" in (span.rationale or ""):
        return _fused_sentence(span.text, after)
    if kind == "OTHER" and _SEX.fullmatch(value):
        return True
    if kind == "LOCATION":
        words = value.split()
        return value.islower() and bool(words) and all(w in _GENERIC_PLACE for w in words)
    return False
