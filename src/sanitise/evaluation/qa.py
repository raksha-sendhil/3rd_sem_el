"""End-to-end utility: answer accuracy on the QA set.  OWNER: Person D.

Answers are paired with QA items by ``answer.query == item.question``; answers to questions
not in the QA set are ignored.  An answer is correct when every string in ``item.expected``
appears in it (case-insensitive, whitespace-normalised) or its generalised form does: the
protected pipeline answers from rewritten text, so "contact dermatitis" may come back as "a
skin condition".  Generalised forms come from the rewriter's own ladders
(``ontology.generalise``) for the categories medical condition, occupation, location, age and
physical; a form that is only the category's generic fallback ("a medical condition") is not
accepted, since it would match almost any answer.

Accuracy is per pipeline: correct answers / answers given by that pipeline.  ``support`` is
the number of QA items.  A pipeline with no answers scores 0.0.
"""

from __future__ import annotations

import logging

from sanitise.agents.rewriter.ontology import GENERIC, generalise
from sanitise.schemas import PIICategory, QAItem, QAMetrics, RagAnswer

log = logging.getLogger(__name__)
_CATEGORIES = (
    PIICategory.MEDICAL_CONDITION,
    PIICategory.OCCUPATION,
    PIICategory.LOCATION,
    PIICategory.AGE,
    PIICategory.PHYSICAL,
)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _forms(expected: str) -> set[str]:
    forms = {_norm(expected)}
    for category in _CATEGORIES:
        try:
            g = generalise(expected, category)
        except Exception:  # a ladder must never break scoring
            log.debug("generalise failed for %r", expected, exc_info=True)
            continue
        if g and g != GENERIC.get(category):
            forms.add(_norm(g))
    return forms


def _correct(answer: str, expected: list[str]) -> bool:
    text = _norm(answer)
    return all(any(f in text for f in _forms(e)) for e in expected)


def score_qa(answers: list[RagAnswer], qa: list[QAItem]) -> QAMetrics:
    by_question = {item.question: item for item in qa}
    totals = {"baseline": [0, 0], "protected": [0, 0]}  # [correct, asked]
    for a in answers:
        item = by_question.get(a.query)
        if item is None:
            continue
        totals[a.pipeline][1] += 1
        totals[a.pipeline][0] += _correct(a.answer, item.expected)
    acc = {k: (c / n if n else 0.0) for k, (c, n) in totals.items()}
    return QAMetrics(
        baseline_accuracy=acc["baseline"],
        protected_accuracy=acc["protected"],
        support=len(qa),
    )
