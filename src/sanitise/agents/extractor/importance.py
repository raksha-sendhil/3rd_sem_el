"""Agent 2 - importance defaults.  OWNER: Person A (Deterministic Logic).

PLACEHOLDER created by Person C. Replace the table; keep the signature.

``default_importance`` gives the importance of a fact from its relation alone. Agent 2 uses
it when the model's importance is missing or not one of high / medium / low, and Agent 3's
utility metric counts how many ``high`` facts survive the rewrite, so this table decides what
the system treats as the useful content of a record (docs/phase3_plan.md, Person A, item 3).
"""

from __future__ import annotations

from sanitise.schemas import Importance

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
}


def default_importance(relation: str) -> Importance:
    return _TABLE.get(relation, Importance.LOW)
