"""Agent 2 - importance defaults.  OWNER: Person A (Deterministic Logic).

``default_importance`` gives the importance of a fact from its relation alone. Agent 2 uses
it when the model's importance is missing or not one of high / medium / low, and Agent 3's
utility metric counts how many ``high`` facts survive the rewrite, so this table decides what
the system treats as the useful content of a record (docs/phase3_plan.md, Person A, item 3).

For legal matters the high-importance facts are the ones the build spec's Agent 2 coverage
target names: the advice, the legal issue, the court type and the outcome, plus the
documents asked for and the recommended next step (the questions lawyers ask). Who the
parties are is low: it is what the rewrite removes.
"""

from __future__ import annotations

from sanitise_legal.schemas import Importance

_TABLE: dict[str, Importance] = {
    "has_condition": Importance.HIGH,
    "has_symptom": Importance.HIGH,
    "took_medication": Importance.HIGH,
    "had_procedure": Importance.HIGH,
    "symptom_duration": Importance.MEDIUM,
    "age": Importance.MEDIUM,
    "sex": Importance.MEDIUM,
    "treated_at": Importance.MEDIUM,
    "event_date": Importance.MEDIUM,
    "height": Importance.LOW,
    "weight": Importance.LOW,
    # legal matters
    "has_legal_issue": Importance.HIGH,
    "advised": Importance.HIGH,
    "filed_in": Importance.HIGH,
    "outcome": Importance.HIGH,
    "documents_requested": Importance.HIGH,
    "next_step": Importance.HIGH,
    "filed_on": Importance.MEDIUM,
    "claim_amount": Importance.MEDIUM,
    "works_as": Importance.MEDIUM,
    "lives_in": Importance.MEDIUM,
    "opposing_party": Importance.LOW,
    "represented_by": Importance.LOW,
    "employed_by": Importance.LOW,
    "contact": Importance.LOW,
    "identifier": Importance.LOW,
}


def default_importance(relation: str) -> Importance:
    return _TABLE.get(relation, Importance.LOW)
