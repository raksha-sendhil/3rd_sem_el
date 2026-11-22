"""What is financial content and not a personal detail: amounts, terms, rates, years, quarters.

Only identities get replaced (docs/finance_plan.md), so Agent 1 must not take "5,205" for a
date, "£500 pounds" for a body weight or "30-year" for a date of someone's life.
"""

from __future__ import annotations

import re

from sanitise_finance.schemas import PIICategory

C = PIICategory

_NUM = r"\d(?:[\d,]*\d)?(?:\.\d+)?"
_SCALE = r"(?:\s?(?:thousand|lakhs?|lacs?|crores?|million|billion|trillion|mn|bn|cr)\b)"
AMOUNT = re.compile(
    rf"(?:[$£€₹¥]\s?{_NUM}|(?<![A-Za-z])(?:USD|GBP|EUR|INR|Rs\.?)\s?{_NUM}){_SCALE}?(?:\s?/-)?"
    # grouped: "5,205", "1,234,567.89", "1,25,000"; not part of a longer number or code
    r"|(?<![\w,.])\d{1,3}(?:,\d{2})*(?:,\d{3})+(?:\.\d+)?(?!\w|,\d)(?:\s?/-)?"
    rf"|(?<![\w,.]){_NUM}(?:{_SCALE}(?:\s(?:dollars|rupees|euros|pounds))?"
    r"|\s(?:dollars|rupees|euros|pounds sterling)\b|\s?/-)",
    re.IGNORECASE,
)

_COUNT = (
    r"(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen"
    r"|twenty|thirty|forty|sixty|ninety)"
)
_PERIOD = re.compile(
    r"(?:(?:within|up\s+to|at\s+least|a\s+term\s+of|the\s+next|first|last|past|over|after|every)\s+)?"
    rf"{_COUNT}(?:\s*(?:-|to)\s*{_COUNT})?[\s-]*(?:(?:business|working|calendar|banking|equal"
    r"|consecutive)\s+)?(?:days?|weeks?|months?|monthly|years?|quarters?|hours?)(?:\s+(?:term|period|fixed))?",
    re.IGNORECASE,
)
_QUARTER = re.compile(
    r"(?:the\s+)?(?:q[1-4]|h[12]|(?:first|second|third|fourth)\s+quarter)", re.IGNORECASE
)
_PLAIN_NUMBER = re.compile(r"[-+(]?\d{1,3}(?:\.\d+)?\)?%?|[-+(]?\d+\.\d{2}\)?|.*\d\s?%")
_YEAR = r"(?:fy\s?|fiscal\s+(?:year\s+)?|financial\s+year\s+|the\s+year\s+)?(?:199\d|20\d\d)"
_YEARS = re.compile(rf"(?:the\s+)?(?:years\s+)?{_YEAR}(?:(?:\s*(?:,|and|&|to|-|–|/)\s*|\s+){_YEAR})*", re.I)
_UNITS = re.compile(r"(?:usd|gbp|eur|inr|rs|and|to|cr|dr)")
_TIME = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]\.?m\.?)?", re.IGNORECASE)
_NOT_AMOUNT_CATEGORIES = {C.DATE, C.PHONE, C.ID_NUMBER, C.AGE, C.PHYSICAL}


def is_amount(text: str) -> bool:
    """Nothing but amounts ("5,205", "4,318   3,905", "£500")."""
    if not AMOUNT.search(text):
        return False
    rest = re.findall(r"[A-Za-z0-9]+", AMOUNT.sub(" ", text))
    return all(_UNITS.fullmatch(w.lower()) for w in rest)


def is_reporting_year(text: str) -> bool:
    """A year on its own, as a report's columns have ("2019", "FY2016", "2019 and 2018"). A
    year identifies no one (HIPAA Safe Harbor keeps it too), and without it a table's figures
    cannot be told apart. Years before 1990 are still generalised: mostly years of birth."""
    return _YEARS.fullmatch(" ".join(text.split())) is not None


def is_content(category: PIICategory, text: str) -> bool:
    """A span of this category that is really financial content, so no span at all."""
    value = " ".join(text.split()).strip(" .,;:")
    if category in _NOT_AMOUNT_CATEGORIES and is_amount(value):
        return True
    if category is C.LOCATION:
        return _QUARTER.fullmatch(value) is not None
    if category is C.OTHER:
        return re.fullmatch(r"[-+(]?\d{1,3}\.\d{1,2}\)?%?", value) is not None
    if category is C.DATE and _TIME.fullmatch(value):
        return True
    if category is C.DATE:
        return bool(
            _PERIOD.fullmatch(value) or _QUARTER.fullmatch(value) or _PLAIN_NUMBER.fullmatch(value)
        )
    return False
