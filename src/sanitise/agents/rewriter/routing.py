"""Agent 3 - routing policy.  OWNER: Person A (Deterministic Logic).

PLACEHOLDER created by Person C: the default table from docs/phase3_plan.md 6.6, without the
rare-condition and QUASI_COMBO refinements. Replace the body; keep the signature.

``route`` returns exactly one ``RoutingDecision`` per input span, in the same order, with
``start``/``end`` copied from the span. ``replacement`` may be left ``None``: the pipeline
fills placeholders (``allocate_placeholders``) and generalised values (``generalise``).

The real version adds (Person A, item 4):
* MEDICAL_CONDITION: keep if common, generalise if it is on the rare list in ontology.py or
  is a member of a QUASI_COMBO;
* QUASI_COMBO: make sure at least one member of the combination is generalised, so the
  combination no longer narrows the candidate pool.
"""

from __future__ import annotations

from sanitise.schemas import Action, FactTuple, PIICategory, RoutingDecision, SensitiveSpan

C = PIICategory

DEFAULT_ACTION: dict[PIICategory, Action] = {
    C.NAME: Action.PLACEHOLDER,
    C.PHONE: Action.PLACEHOLDER,
    C.EMAIL: Action.PLACEHOLDER,
    C.ID_NUMBER: Action.PLACEHOLDER,
    C.ADDRESS: Action.PLACEHOLDER,
    C.AGE: Action.GENERALISE,
    C.DATE: Action.GENERALISE,
    C.LOCATION: Action.GENERALISE,
    C.OCCUPATION: Action.GENERALISE,
    C.PHYSICAL: Action.GENERALISE,
    C.QUASI_COMBO: Action.GENERALISE,
    C.MEDICAL_CONDITION: Action.KEEP,
    C.SYMPTOM: Action.KEEP,
    C.RELATION: Action.KEEP,
    C.OTHER: Action.PLACEHOLDER,
}


def route(spans: list[SensitiveSpan], facts: list[FactTuple]) -> list[RoutingDecision]:
    return [
        RoutingDecision(
            span_text=span.text,
            category=span.category,
            action=DEFAULT_ACTION[span.category],
            reason=f"default policy for {span.category.value}",
            start=span.start,
            end=span.end,
        )
        for span in spans
    ]
