"""Agent 2 - subjects and fact normalisation.  OWNER: Person A (Deterministic Logic).

* ``subjects``: every person the document talks about, each with a stable id and its
  mentions as written:

  - ``writer``: the author;
  - ``writer.<relation>`` for a relative of the writer (``my son`` -> ``writer.son``);
  - ``client``, ``opposing_party``, ``witness``: the parties of a legal matter, from the
    role rules/roles.py tagged on NAME spans ("Client: Rohan Mehta", "Respondent: Sunita
    Kale"), with every mention of the same name (including a lone surname);
  - ``person_<n>`` for any other named person.

  Agent 2's prompt receives this list, so the model attaches each fact to the right person.
* ``normalise_facts``: map relation synonyms onto ``schemas.RELATIONS`` ("lives in",
  "resides_in" -> ``lives_in``; "advice" -> ``advised``; "court" -> ``filed_in``), trim
  values, and merge duplicates (same subject, relation and value, ignoring case and spacing).
  Unknown relations become ``other``. Facts about subjects that are not in the list are kept:
  the advice in a matter is often attached to "the matter" rather than to a person.
"""

from __future__ import annotations

import re

from sanitise.rules.roles import name_words, role_of
from sanitise.schemas import RELATIONS, FactTuple, PIICategory, SensitiveSpan, Subject

_KIN = re.compile(r"(?:my|his|her|our|their)\s+(.+)$", re.IGNORECASE)
_ROLE_SUBJECT = {
    "client": ("client", "the client"),
    "party": ("opposing_party", "the opposing party"),
    "witness": ("witness", "the witness"),
}


def _add(found: list[Subject], sid: str, label: str, mention: str) -> None:
    for s in found:
        if s.id == sid:
            if mention not in s.mentions:
                s.mentions.append(mention)
            return
    found.append(Subject(id=sid, label=label, mentions=[mention]))


def subjects(text: str, spans: list[SensitiveSpan]) -> list[Subject]:
    found = [Subject(id="writer", label="the writer", mentions=[])]
    names = [s for s in spans if s.category is PIICategory.NAME]

    # which subject each name belongs to: the role of any mention of the same person
    owner: dict[tuple[str, ...], tuple[str, str]] = {}
    for span in names:
        role = role_of(span)
        key = tuple(name_words(span.text))
        if role and key and key not in owner:
            owner[key] = _ROLE_SUBJECT[role]
    others = 0
    for span in names:
        key = tuple(name_words(span.text))
        if not key:
            continue
        if key not in owner and len(key) == 1:
            full = [owner[k] for k in owner if len(k) > 1 and key[0] in k]
            if len(set(full)) == 1:
                owner[key] = full[0]
        if key not in owner:
            others += 1
            owner[key] = (f"person_{others}", f"person {others}")
        sid, label = owner[key]
        _add(found, sid, label, span.text)

    for span in spans:
        if span.category is not PIICategory.RELATION:
            continue
        match = _KIN.match(span.text.strip())
        kin = (match.group(1) if match else span.text).strip().lower().replace(" ", "_")
        _add(found, f"writer.{kin}", f"the writer's {kin.replace('_', ' ')}", span.text)
    return found


# Relation names a model tends to produce instead of the vocabulary's own.
SYNONYMS: dict[str, str] = {
    "lives": "lives_in",
    "resides_in": "lives_in",
    "located_in": "lives_in",
    "city": "lives_in",
    "location": "lives_in",
    "occupation": "works_as",
    "job": "works_as",
    "profession": "works_as",
    "employer": "employed_by",
    "works_for": "employed_by",
    "works_at": "employed_by",
    "legal_issue": "has_legal_issue",
    "issue": "has_legal_issue",
    "dispute": "has_legal_issue",
    "matter_type": "has_legal_issue",
    "case_type": "has_legal_issue",
    "offence": "has_legal_issue",
    "court": "filed_in",
    "forum": "filed_in",
    "filed_before": "filed_in",
    "court_type": "filed_in",
    "filing_date": "filed_on",
    "filed": "filed_on",
    "date_filed": "filed_on",
    "amount": "claim_amount",
    "claim": "claim_amount",
    "amount_claimed": "claim_amount",
    "cheque_amount": "claim_amount",
    "opponent": "opposing_party",
    "opposite_party": "opposing_party",
    "counterparty": "opposing_party",
    "respondent": "opposing_party",
    "defendant": "opposing_party",
    "lawyer": "represented_by",
    "counsel": "represented_by",
    "advocate": "represented_by",
    "advice": "advised",
    "advice_given": "advised",
    "recommendation": "advised",
    "recommended": "advised",
    "documents": "documents_requested",
    "documents_needed": "documents_requested",
    "documents_to_collect": "documents_requested",
    "result": "outcome",
    "status": "outcome",
    "decision": "outcome",
    "judgment": "outcome",
    "recommended_step": "next_step",
    "next_action": "next_step",
    "condition": "has_condition",
    "diagnosis": "has_condition",
    "symptom": "has_symptom",
    "medication": "took_medication",
    "date": "event_date",
    "phone": "contact",
    "email": "contact",
    "id": "identifier",
}


def canonical_relation(relation: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", relation.lower()).strip("_")
    if key in RELATIONS:
        return key
    return SYNONYMS.get(key, "other")


def normalise_facts(facts: list[FactTuple], subjects: list[Subject]) -> list[FactTuple]:
    out: list[FactTuple] = []
    seen: set[tuple[str, str, str]] = set()
    for fact in facts:
        relation = canonical_relation(fact.relation)
        value = " ".join(fact.value.split()).strip(" .,;")
        key = (fact.subject, relation, value.lower())
        if not value or key in seen:
            continue
        seen.add(key)
        out.append(fact.model_copy(update={"relation": relation, "value": value}))
    return out
