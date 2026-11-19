"""Is a document a clinical record (a case sheet, clinic note or discharge summary)?  OWNER: C.

Everything the pipeline does differently for a clinical record hangs on this; any other document
(a patient's message, an email) goes through exactly as it would without case-file support.

* Agents 1 and 2 see the text and ask ``is_case_file(text)``: three or more different labelled
  clinical sections (C/O, H/O, Diagnosis, Plan, MRN ...).
* Agent 3 never sees the text. It knows from Agent 2's subjects (``patient``) and sets
  ``CASE_FILE`` for the length of its rewrite (``case_file(...)``), so the generalisation and
  value rewriting it calls can ask too.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_SECTION = re.compile(
    r"^[ \t]*(c/o|h/o|o/e|k/c/o|chief complaints?|presenting complaints?|complaints?"
    r"|history of present(?:ing)? illness|hpi|past (?:medical )?history|pmh|family history"
    r"|examination|exam|vitals|vital signs|investigations?|inv|labs?|lab results"
    r"|impression|imp|diagnosis|provisional diagnosis|final diagnosis|assessment|plan|advice"
    r"|rx|treatment|medications?|allerg(?:y|ies)|course in (?:the )?hospital|hospital course"
    r"|follow[- ]up|mrn|uhid|ip no|op no|dob|age/sex|date of (?:visit|admission|discharge)"
    r"|admitted|discharged|consultant|referred by|patient(?: name)?)\b[^:\n]{0,15}:",
    re.IGNORECASE | re.MULTILINE,
)

CASE_FILE: ContextVar[bool] = ContextVar("case_file", default=False)


def is_case_file(text: str) -> bool:
    """Three or more different labelled clinical sections."""
    return len({m.group(1).lower() for m in _SECTION.finditer(text)}) >= 3


@contextmanager
def case_file(on: bool) -> Iterator[None]:
    """``CASE_FILE`` is ``on`` inside the block (Agent 3's rewrite of one document)."""
    token = CASE_FILE.set(on)
    try:
        yield
    finally:
        CASE_FILE.reset(token)
