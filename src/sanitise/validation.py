"""Boundary checks that Pydantic cannot do alone because they need the document text."""

from __future__ import annotations

import re
from collections.abc import Iterable

from sanitise.errors import OffsetOutOfRange, Stage
from sanitise.schemas import SensitiveSpan

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip().casefold()


def check_span_offsets(text: str, spans: Iterable[SensitiveSpan], *, stage: Stage) -> None:
    """Every span with offsets must lie inside ``text`` and its ``text`` must match the slice.

    The match ignores case and whitespace runs (the spec's own example labels
    ``"My son"`` at 0-6 as ``"my son"``). Spans without offsets are skipped.
    """
    for i, span in enumerate(spans):
        if span.start is None or span.end is None:
            continue
        if not 0 <= span.start < span.end <= len(text):
            raise OffsetOutOfRange(
                f"span {i} {span.category.value} {span.text!r} has chars {span.start}-{span.end} "
                f"but the document has {len(text)} characters",
                stage=stage,
            )
        actual = text[span.start : span.end]
        if _norm(actual) != _norm(span.text):
            raise OffsetOutOfRange(
                f"span {i} {span.category.value} says {span.text!r} but chars "
                f"{span.start}-{span.end} are {actual!r}",
                stage=stage,
            )
