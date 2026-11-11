"""Agent 3 - routing policy.  OWNER: Person A (Deterministic Logic).

``route`` returns exactly one ``RoutingDecision`` per input span, in the same order, with
``start``/``end`` copied from the span. The policy is the legal build spec's table
(docs/phase3_plan.md 6.6 for the medical categories):

* placeholder: NAME, PHONE, EMAIL, ID_NUMBER, FINANCIAL, CASE_NUMBER, ADDRESS,
  MATTER_CODENAME (``[CLIENT_1]``, ``[CASE_1]``, ``[MATTER_1]``, ...)
* generalise:  ORGANIZATION, COURT, LOCATION, DATE, MONEY, AGE, OCCUPATION, PHYSICAL
* keep:        RELATION, SYMPTOM, and LEGAL_ISSUE / MEDICAL_CONDITION unless they are rare
  (``ontology.is_rare``), since the legal issue is what lawyers search for
* QUASI_COMBO: every member of the combination is generalised two steps instead of one; if
  no member would otherwise change, the first keepable member is generalised, so the
  combination no longer narrows the pool to one client.
* In the legal domain a document that names three or more of a person's age, occupation,
  location and dates is treated as a combination even when the LLM pass reported none (the
  rules-only fallback, or a model that missed it): those spans climb two steps. "in their
  50s" + "a doctor" + "2024" still narrows a client table to one person; "middle-aged" +
  "a professional" + "2024" does not.

``replacement`` is filled here only for two-step generalisations; the pipeline fills
placeholders (``allocate_placeholders``) and one-step generalisations (``generalise``).
A NAME span's party role (``role=client``, rules/roles.py) is copied into ``reason``, which
is where ``allocate_placeholders`` reads it.
"""

from __future__ import annotations

import re

from sanitise_legal.agents.rewriter.ontology import generalise, is_rare
from sanitise_legal.domain import is_legal
from sanitise_legal.rules.roles import role_of
from sanitise_legal.schemas import Action, FactTuple, PIICategory, RoutingDecision, SensitiveSpan

C = PIICategory

DEFAULT_ACTION: dict[PIICategory, Action] = {
    C.NAME: Action.PLACEHOLDER,
    C.PHONE: Action.PLACEHOLDER,
    C.EMAIL: Action.PLACEHOLDER,
    C.ID_NUMBER: Action.PLACEHOLDER,
    C.FINANCIAL: Action.PLACEHOLDER,
    C.CASE_NUMBER: Action.PLACEHOLDER,
    C.ADDRESS: Action.PLACEHOLDER,
    C.MATTER_CODENAME: Action.PLACEHOLDER,
    C.ORGANIZATION: Action.GENERALISE,
    C.COURT: Action.GENERALISE,
    C.MONEY: Action.GENERALISE,
    C.AGE: Action.GENERALISE,
    C.DATE: Action.GENERALISE,
    C.LOCATION: Action.GENERALISE,
    C.OCCUPATION: Action.GENERALISE,
    C.PHYSICAL: Action.GENERALISE,
    C.QUASI_COMBO: Action.GENERALISE,
    C.LEGAL_ISSUE: Action.KEEP,
    C.MEDICAL_CONDITION: Action.KEEP,
    C.SYMPTOM: Action.KEEP,
    C.RELATION: Action.KEEP,
    C.OTHER: Action.PLACEHOLDER,
}
_RARE_CHECKED = {C.LEGAL_ISSUE, C.MEDICAL_CONDITION}
# Kept categories that may be generalised to break a combination. RELATION ("my client's
# wife") and SYMPTOM carry no searchable attribute, so changing them would not help.
_BREAKABLE = (C.OCCUPATION, C.LOCATION, C.AGE, C.DATE, C.LEGAL_ISSUE, C.MEDICAL_CONDITION)
# Categories whose co-occurrence makes an implicit combination (legal domain).
_IMPLICIT = (C.AGE, C.OCCUPATION, C.LOCATION, C.DATE)

# How a combination lists its parts: "cardiologist + Pune + land dispute + 2024"
_PART_SPLIT = re.compile(r"\s*(?:\+|&|;|,|\band\b|\bwith\b|\bplus\b|\bin\b)\s*", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def _norm(text: str) -> str:
    return _NON_ALNUM.sub(" ", text.lower()).strip()


def _default(span: SensitiveSpan) -> RoutingDecision:
    action = DEFAULT_ACTION[span.category]
    reason = f"default policy for {span.category.value}"
    if span.category in _RARE_CHECKED and is_rare(span.text):
        action, reason = Action.GENERALISE, f"rare {span.category.value}: would narrow the pool"
    if span.category is C.NAME and (role := role_of(span)):
        reason += f" role={role}"
    return RoutingDecision(
        span_text=span.text,
        category=span.category,
        action=action,
        reason=reason,
        start=span.start,
        end=span.end,
    )


def _members(combo: SensitiveSpan, spans: list[SensitiveSpan]) -> list[int]:
    """Indices of the positional spans the combination is made of."""
    parts = [p for p in (_norm(x) for x in _PART_SPLIT.split(combo.text)) if p]
    whole = f" {_norm(combo.text)} "
    found = []
    for i, span in enumerate(spans):
        if span.start is None or span.category is C.QUASI_COMBO:
            continue
        text = _norm(span.text)
        if text and (
            f" {text} " in whole or any(p == text or f" {p} " in f" {text} " for p in parts)
        ):
            found.append(i)
    return found


def route(spans: list[SensitiveSpan], facts: list[FactTuple]) -> list[RoutingDecision]:
    decisions = [_default(span) for span in spans]
    for combo in (s for s in spans if s.category is C.QUASI_COMBO):
        members = _members(combo, spans)
        if not members:
            continue
        if all(decisions[i].action is Action.KEEP for i in members):
            breakable = [i for i in members if spans[i].category in _BREAKABLE]
            if breakable:
                i = breakable[0]
                decisions[i].action = Action.GENERALISE
        for i in members:
            d = decisions[i]
            if d.action is Action.GENERALISE:
                d.replacement = generalise(d.span_text, d.category, steps=2)
                d.reason = f"member of QUASI_COMBO '{combo.text}': generalised two steps"
    if is_legal():
        _break_implicit(spans, decisions)
    return decisions


def _break_implicit(spans: list[SensitiveSpan], decisions: list[RoutingDecision]) -> None:
    members = [
        i
        for i, s in enumerate(spans)
        if s.start is not None
        and s.category in _IMPLICIT
        and decisions[i].action is Action.GENERALISE
    ]
    if len({spans[i].category for i in members}) < 3:
        return
    for i in members:
        d = decisions[i]
        if d.reason.startswith("member of QUASI_COMBO"):
            continue
        d.replacement = generalise(d.span_text, d.category, steps=2)
        d.reason = "age, occupation, location and dates together: generalised two steps"
