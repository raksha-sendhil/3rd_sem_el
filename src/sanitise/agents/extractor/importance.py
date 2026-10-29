"""Agent 2 - importance defaults.  OWNER: Person A (Deterministic Logic).

``default_importance`` gives the importance of a fact from its relation alone. Agent 2 uses
it when the model's importance is missing or not one of high / medium / low, and Agent 3's
utility metric counts how many ``high`` facts survive the rewrite, so this table decides what
the system treats as the useful content of a record (docs/phase3_plan.md, Person A, item 3).

The table follows one rule: **high** is the clinical content a legitimate user asks about
(conditions, symptoms, medication, procedures); **medium** is context that changes the
clinical reading but also narrows who the person is (age, sex, where they live, their job,
how long it has lasted, when it happened); **low** is detail that mostly identifies (contact
details, identifiers, the hospital, height and weight). Relation names are normalised first,
so a synonym such as ``"diagnosed_with"`` gets the importance of ``has_condition``.
"""

from __future__ import annotations

from sanitise.agents.extractor.canonical import normalise_relation
from sanitise.schemas import RELATIONS, Importance

H, M, L = Importance.HIGH, Importance.MEDIUM, Importance.LOW

IMPORTANCE: dict[str, Importance] = {
    "has_condition": H,
    "has_symptom": H,
    "took_medication": H,
    "had_procedure": H,
    "progress": H,
    "test_result": H,
    "allergy": H,
    "assessment": H,
    "vital_sign": M,
    "plan": M,
    "symptom_duration": M,
    "age": M,
    "sex": M,
    "lives_in": M,
    "works_as": M,
    "event_date": M,
    "relation_to": M,
    "studies_at": L,
    "treated_at": L,
    "height": L,
    "weight": L,
    "contact": L,
    "identifier": L,
    "other": L,
}
assert set(IMPORTANCE) == set(RELATIONS), "every relation in the vocabulary needs an importance"


def default_importance(relation: str) -> Importance:
    return IMPORTANCE[normalise_relation(relation)]
