"""End-to-end utility: answer accuracy on the QA set.  OWNER: Person D.

Answers are paired with QA items by ``answer.query == item.question``. An answer is correct
when every string in ``item.expected`` appears in it, ignoring case, punctuation and spacing
("builder-buyer agreement" matches "builder buyer agreement"). The legal QA set's expected
phrases are advice and document phrases, which the protected pipeline keeps verbatim, so no
generalised form is needed for them.
"""

from __future__ import annotations

import re

from sanitise.schemas import QAItem, QAMetrics, RagAnswer

_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def _norm(text: str) -> str:
    return " " + _NON_ALNUM.sub(" ", text.lower()).strip() + " "


def _correct(answer: str, expected: list[str]) -> bool:
    text = _norm(answer)
    return all(_norm(e) in text for e in expected)


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
