"""Agent 3 - routing policy.  OWNER: Person A (Deterministic Logic).

``route(spans, facts)`` returns exactly one ``RoutingDecision`` per input span, in the same
order, with ``start``/``end`` copied from the span, following docs/phase3_plan.md 6.6:

=================================  ===========  ============================================
Category                           Action       Notes
=================================  ===========  ============================================
NAME, PHONE, EMAIL, ID_NUMBER,     placeholder  token filled later by ``allocate_placeholders``
ADDRESS, OTHER
AGE, DATE, LOCATION, OCCUPATION,   generalise   one step up the ladder (``ontology``)
PHYSICAL
MEDICAL_CONDITION                  keep         generalised instead when it is rare
                                                (``ontology.is_rare``) or part of a QUASI_COMBO
SYMPTOM, RELATION                  keep         generalised only to break a QUASI_COMBO
QUASI_COMBO                        generalise   no position of its own; broken through members
=================================  ===========  ============================================

**QUASI_COMBO.** A combination is broken by generalising its members, the positional spans it
is made of. Members are found by text (a span whose words appear in the combination, or a part
of the combination that appears in the span: ``lighthouse keeper + Port Isaac``) or, when the
combination only describes its parts (``child age + named school + small town``), by the
categories it names. Members in a category that is generalised by default climb **two** steps
instead of one (plan section 9); a condition in a combination is generalised one step (it is
kept otherwise), so the record still says what kind of illness it was. If no member would be
generalised by default (a combination of symptoms, say), the one with the least important fact
behind it is generalised, preferring a condition, then a symptom, then a relation, so the
combination no longer narrows the candidate pool.

For generalised spans the ``replacement`` is filled here (so the two-step members get their
two-step phrase); placeholders are left to ``allocate_placeholders``. ``facts`` decide which
member of an all-keep combination gives way: the one whose fact matters least.
"""

from __future__ import annotations

import re

from sanitise.agents.rewriter import ontology
from sanitise.schemas import (
    Action,
    FactTuple,
    Importance,
    PIICategory,
    RoutingDecision,
    SensitiveSpan,
)

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
assert set(DEFAULT_ACTION) == set(PIICategory), "every category needs a default action"

# words in a descriptive combination -> the categories of its members
_COMBO_WORDS: list[tuple[str, tuple[PIICategory, ...]]] = [
    (r"\bage\b|\baged?\b|years? old|elderly|\bchild\b", (C.AGE,)),
    (r"\bdates?\b|timeline|onset|\bday\b|\bmonths?\b|\bweeks?\b", (C.DATE,)),
    (
        r"town|city|village|island|location|residence|place|school|hospital|institution"
        r"|clinic|desk|district|region|area|address",
        (C.LOCATION, C.ADDRESS),
    ),
    (r"occupation|\bjob\b|profession|athlete|player|employ|career|\bwork", (C.OCCUPATION,)),
    (
        r"condition|diagnos|disease|cancer|injury|arrhythmia|disorder|illness|syndrome",
        (C.MEDICAL_CONDITION,),
    ),
    (r"symptom", (C.SYMPTOM,)),
    (r"height|weight|build|\btall\b|physical", (C.PHYSICAL,)),
]
# which kept member gives way first when a combination has no generalised member
_BREAK_ORDER = [C.MEDICAL_CONDITION, C.SYMPTOM, C.RELATION]
_IMPORTANCE_RANK = {Importance.LOW: 0, Importance.MEDIUM: 1, Importance.HIGH: 2}


def _norm(text: str) -> str:
    return " " + " ".join(re.findall(r"[a-z0-9]+", text.lower())) + " "


def combo_members(combo: SensitiveSpan, spans: list[SensitiveSpan]) -> list[int]:
    """Indexes of the positional spans a QUASI_COMBO is made of."""
    combo_text = _norm(combo.text)
    parts = [
        _norm(p)
        for p in re.split(r"\s*(?:\+|&|;|,|\(|\)|\band\b|\bwith\b|\bplus\b)\s*", combo.text)
        if len(p.strip()) >= 4
    ]
    positional = [
        i for i, s in enumerate(spans) if s.start is not None and s.category is not C.QUASI_COMBO
    ]
    members = [
        i
        for i in positional
        if _norm(spans[i].text) in combo_text or any(p in _norm(spans[i].text) for p in parts)
    ]
    if members:
        return members
    wanted = {
        cat
        for pattern, cats in _COMBO_WORDS
        if re.search(pattern, combo.text, re.IGNORECASE)
        for cat in cats
    }
    return [i for i in positional if spans[i].category in wanted]


def _fact_importance(span: SensitiveSpan, facts: list[FactTuple]) -> int:
    """Highest importance of a fact that mentions this span (``-1`` when none does)."""
    needle = _norm(span.text)
    ranks = [_IMPORTANCE_RANK[f.importance] for f in facts if needle in _norm(f.value)]
    return max(ranks, default=-1)


def route(spans: list[SensitiveSpan], facts: list[FactTuple]) -> list[RoutingDecision]:
    actions = [DEFAULT_ACTION[s.category] for s in spans]
    reasons = [f"default policy for {s.category.value}" for s in spans]
    levels = [1] * len(spans)

    for i, span in enumerate(spans):
        if span.category is C.MEDICAL_CONDITION:
            if ontology.is_rare(span.text):
                actions[i] = Action.GENERALISE
                reasons[i] = "rare condition: it would narrow the candidate pool"
            else:
                reasons[i] = "common condition: kept for clinical usefulness"

    for combo in (s for s in spans if s.category is C.QUASI_COMBO):
        members = combo_members(combo, spans)
        if not members:
            continue
        for i in members:
            if spans[i].category is C.MEDICAL_CONDITION:
                actions[i] = Action.GENERALISE
        if not any(actions[i] is not Action.KEEP for i in members):
            keepers = [i for i in members if spans[i].category in _BREAK_ORDER]
            changes = [
                i
                for i in keepers
                if ontology.generalise(spans[i].text, spans[i].category)
                != ontology.GENERIC[spans[i].category]
            ] or keepers
            if not changes:
                continue
            pick = min(
                changes,
                key=lambda i: (
                    _fact_importance(spans[i], facts),
                    _BREAK_ORDER.index(spans[i].category),
                    i,
                ),
            )
            actions[pick] = Action.GENERALISE
            reasons[pick] = f"generalised to break the combination '{combo.text}'"
            continue  # a kept detail gives way by one step only
        for i in members:
            if DEFAULT_ACTION[spans[i].category] is Action.GENERALISE:
                levels[i] = 2
                reasons[i] = f"member of the combination '{combo.text}': generalised two steps"
            elif actions[i] is Action.GENERALISE:
                reasons[i] = f"member of the combination '{combo.text}': generalised"

    decisions = []
    for span, action, reason, level in zip(spans, actions, reasons, levels, strict=True):
        replacement = None
        if action is Action.GENERALISE:
            replacement = ontology.generalise(span.text, span.category, level=level)
        decisions.append(
            RoutingDecision(
                span_text=span.text,
                category=span.category,
                action=action,
                replacement=replacement,
                reason=reason,
                start=span.start,
                end=span.end,
            )
        )
    return decisions
