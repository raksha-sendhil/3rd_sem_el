"""End-to-end utility: answer accuracy on the QA set.  OWNER: Person D.

PLACEHOLDER created by Person C (a working first version). Refine the body; keep the
signature (docs/phase3_plan.md 6.2).

Answers are paired with QA items by ``answer.query == item.question``. An answer is correct
when every string in ``item.expected`` appears in it (case-insensitive). The real version
should also accept the generalised form of an expected string, since the protected pipeline
answers from rewritten text.
"""

from __future__ import annotations

from sanitise.schemas import QAItem, QAMetrics, RagAnswer


def _correct(answer: str, expected: list[str]) -> bool:
    text = answer.lower()
    return all(e.lower() in text for e in expected)


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
