"""Is a document a clinical record (a case sheet, clinic note or discharge summary)?  OWNER: C.

In finance the answer is always no (see ``is_case_file``); the switch stays so that the code
copied from the healthcare pipeline keeps its shape.

Everything the pipeline does differently for a clinical record hangs on this; any other document
(a patient's message, an email) goes through exactly as it would without case-file support.

* Agents 1 and 2 see the text and ask ``is_case_file(text)``.
* Agent 3 never sees the text. It knows from Agent 2's subjects (``patient``) and sets
  ``CASE_FILE`` for the length of its rewrite (``case_file(...)``), so the generalisation and
  value rewriting it calls can ask too.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

CASE_FILE: ContextVar[bool] = ContextVar("case_file", default=False)


def is_case_file(text: str) -> bool:
    """Never, in finance: a form's "DOB:", "Plan:" and "Assessment:" labels would pass for
    clinical sections, and the form would be rewritten as a patient's case summary."""
    return False


@contextmanager
def case_file(on: bool) -> Iterator[None]:
    """``CASE_FILE`` is ``on`` inside the block (Agent 3's rewrite of one document)."""
    token = CASE_FILE.set(on)
    try:
        yield
    finally:
        CASE_FILE.reset(token)
