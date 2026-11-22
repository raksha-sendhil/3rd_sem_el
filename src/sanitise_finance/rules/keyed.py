"""Identifiers announced by their own label.  OWNER: Person A.

The recognisers know the formats they know. A value that the text itself calls a postal code, a
SWIFT code or a password is an identifier whatever it looks like, and Agent 2 copies what Agent 1
missed into the facts. ``keyed_identifiers`` adds those values after the merge, where no span
already covers them.
"""

from __future__ import annotations

import re

from sanitise_finance.schemas import PIICategory, SensitiveSpan

_GAP = r"[\s*_]*(?:(?:is|was|of)\s+|[:#=-]\s*)?[\s*_\"']*"
_FOR = r"(?:\s+for\s+(?:(?:the|this)\s+)?\w+)?"
_RULES = [
    # a label, then a value with a digit in it
    re.compile(
        r"\b(?:swift(?:[\s/-]*bic)?(?:\s+code)?|bic(?:\s+code)?|iban|ifsc(?:\s+code)?"
        r"|routing\s+(?:number|no\.?)|sort\s+code|postal\s+code|post\s?code|zip(?:\s*code)?"
        r"|pin\s*code|cvv2?|cvc|passport\s+(?:number|no\.?)"
        r"|(?:policy|licen[cs]e|account|card|loan|routing|beneficiary|member(?:ship)?|insurance|registration|claim)"
        rf"\s+(?:number|no\.?|id|code)){_FOR}{_GAP}"
        r"(?P<value>(?=[A-Za-z/-]*\d)[A-Za-z0-9][A-Za-z0-9/-]{2,33})(?![A-Za-z0-9])",
        re.IGNORECASE,
    ),
    # a label, then a secret that may be all letters
    re.compile(
        rf"\b(?:password|passcode)(?:\s+is\s+|[\s*_]*[:=]){_GAP}"
        r"(?P<value>[^\s\"'*<>]{4,40}?)(?=[.,;]?(?:[\s\"'*<>]|$))",
        re.IGNORECASE,
    ),
]


def keyed_identifiers(text: str, spans: list[SensitiveSpan]) -> list[SensitiveSpan]:
    """The labelled values in ``text`` that no detected span covers."""
    taken = [(s.start, s.end) for s in spans if s.start is not None and s.end is not None]
    found: list[SensitiveSpan] = []
    for rule in _RULES:
        for m in rule.finditer(text):
            start, end = m.span("value")
            if any(a < end and start < b for a, b in taken):
                continue
            taken.append((start, end))
            found.append(
                SensitiveSpan(
                    text=text[start:end],
                    category=PIICategory.ID_NUMBER,
                    start=start,
                    end=end,
                    method="rule",
                    rationale="keyed: labelled as an identifier in the text",
                )
            )
    return found
