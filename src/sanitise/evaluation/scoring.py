"""Stage 7 - scoring against the gold set.  OWNER: Person D (Data and Evaluation).

PLACEHOLDER created by Person C. ``sanitise score`` and ``POST /v1/score``
already call this function; until it is implemented they report "not
implemented" (exit code 3 / HTTP 501). Keep the signature.

Match rule (spec 3.2): a predicted span is correct if its character range
overlaps the gold span and the category matches. Quasi-identifiers (no
offsets) are matched on category alone. Recall is the primary figure.

``gold`` arrives already loaded and validated by ``sanitise.results.load_gold``.
List order is not guaranteed: pair ``pred`` and ``gold`` documents by ``doc_id``.
"""

from __future__ import annotations

from sanitise.schemas import DetectorOutput, SpanMetrics


def score(pred: list[DetectorOutput], gold: list[DetectorOutput]) -> SpanMetrics:
    """Overlap match on start/end plus category equality.
    Quasi-identifiers matched on category alone."""
    # TODO(Person D): implement precision / recall / F1, overall and per category.
    raise NotImplementedError("score() is not implemented yet (Person D)")
