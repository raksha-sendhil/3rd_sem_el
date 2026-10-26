"""Agent 2 - subjects and fact normalisation.  OWNER: Person A (Deterministic Logic).

PLACEHOLDER created by Person C so the pipeline runs end to end from the first push.
Replace the bodies; keep the signatures (checked by tests/contract/test_phase3_interfaces.py).

What the real version does (docs/phase3_plan.md, section 4, Person A, items 1-2):

* ``subjects``: find every person the document talks about and give each a stable id -
  ``writer`` (the author), ``writer.<relation>`` for a relative of the writer
  (``writer.son``, ``writer.wife``), ``person_<n>`` for anyone else - with every mention
  as written in the text ("my son", "he", "Naipaul"). Agent 2's prompt receives this list,
  so the model attaches each fact to the right person.
* ``normalise_facts``: map relation synonyms onto ``schemas.RELATIONS`` ("lives in",
  "resides_in" -> ``lives_in``), drop facts whose subject is not in ``subjects``, merge
  duplicates (same subject, relation and value, ignoring case and spacing), trim values.
"""

from __future__ import annotations

import re

from sanitise.schemas import RELATIONS, FactTuple, PIICategory, SensitiveSpan, Subject

_KIN = re.compile(r"(?:my|his|her|our|their)\s+(.+)$", re.IGNORECASE)


def subjects(text: str, spans: list[SensitiveSpan]) -> list[Subject]:
    """The writer plus one subject per RELATION span (``My son`` -> ``writer.son``)."""
    found = [Subject(id="writer", label="the writer", mentions=[])]
    seen = {"writer"}
    for span in spans:
        if span.category is not PIICategory.RELATION:
            continue
        match = _KIN.match(span.text.strip())
        kin = (match.group(1) if match else span.text).strip().lower().replace(" ", "_")
        sid = f"writer.{kin}"
        if sid in seen:
            next(s for s in found if s.id == sid).mentions.append(span.text)
            continue
        seen.add(sid)
        found.append(
            Subject(id=sid, label=f"the writer's {kin.replace('_', ' ')}", mentions=[span.text])
        )
    return found


def normalise_facts(facts: list[FactTuple], subjects: list[Subject]) -> list[FactTuple]:
    """Unknown relations become ``other``; exact duplicates are dropped."""
    out: list[FactTuple] = []
    seen: set[tuple[str, str, str]] = set()
    for fact in facts:
        relation = fact.relation if fact.relation in RELATIONS else "other"
        value = " ".join(fact.value.split())
        key = (fact.subject, relation, value.lower())
        if not value or key in seen:
            continue
        seen.add(key)
        out.append(fact.model_copy(update={"relation": relation, "value": value}))
    return out
