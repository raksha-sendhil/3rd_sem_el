"""Agent 2 - fact-extraction prompt.  OWNER: Person C (Model, Platform and Integration).

One constrained-JSON call per document (docs/phase3_plan.md, Person C, item 2). The model gets
the document, the people in it (``subjects``, from Person A - every fact must name one of
their ids) and the sensitive details Agent 1 found (hints, so identifiers are not missed), and
returns ``FactTuple``s whose ``relation`` is one of ``schemas.RELATIONS``.

Decoding is constrained to ``LLMFactList``: ``relation`` and ``importance`` are enums, so the
model cannot invent a relation name. The pipeline then runs Person A's ``normalise_facts`` and
drops facts whose subject is unknown.

Design notes:
* Identifying details (names, contact details, IDs, places) are extracted as facts too:
  Agent 3 needs to know about them to write a complete record, and replaces them before the
  reconstruction model ever sees them.
* ``value`` copies the document's wording, so later stages can match it against the text.
* The few-shot examples are written for this prompt. They must never be taken from
  ``data/fixtures`` or ``data/gold``, because the evaluation runs on those documents
  (``tests/prompts/test_prompts.py`` fails if a gold phrase appears here).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sanitise.llm.base import LLMClient
from sanitise.schemas import RELATIONS, FactTuple, Importance, SensitiveSpan, Subject

Relation = Literal[RELATIONS]  # type: ignore[valid-type]


class LLMFact(BaseModel):
    subject: str = Field(min_length=1)
    relation: Relation
    value: str = Field(min_length=1, max_length=300)
    importance: Literal["high", "medium", "low"]


class LLMFactList(BaseModel):
    """The schema handed to Ollama. Titled "FactList" (MockLLM scripts use that name)."""

    model_config = ConfigDict(title="FactList")
    facts: list[LLMFact]


SYSTEM_PROMPT = """\
You turn one private record (a patient's message to a doctor, or a workplace email) into a \
list of facts. Each fact is {"subject", "relation", "value", "importance"}.

subject: the id of the person the fact is about, from SUBJECTS. "writer" is the person who \
wrote the record; "writer.<relative>" is one of their relatives; "person_<n>" is anyone else.

relation: exactly one of these names:
- age, sex: the person's age or sex
- lives_in: a town, city or address where the person lives
- works_as: the person's job or role; studies_at: a school or university they attend
- treated_at: a named hospital, clinic or doctor's practice treating them
- has_condition: a named diagnosis, disease or clinical finding (also a suspected one)
- has_symptom: a symptom or complaint (fever, cough, pain somewhere, swelling)
- symptom_duration: how long a symptom has lasted ("three weeks")
- took_medication, had_procedure: a drug they take; an operation or test they had
- event_date: a specific date of an event in their life (diagnosis, admission, leave)
- relation_to: how the person is related to the writer ("son", "wife")
- height, weight: body measurements
- contact: a phone number or email address; identifier: a name or an ID number
- other: anything else that describes the person

value: the detail in a few words, copied from the record's own wording. Do not add words \
that are not in the record.

importance: "high" for has_condition, has_symptom, took_medication and had_procedure; \
"medium" for age, sex, treated_at, event_date and symptom_duration; "low" for everything else.

Rules:
- One fact per detail. Every name, age, place, job, date, contact detail and ID in the record \
is a fact, and so is every condition, symptom, medication and procedure.
- Only facts about the people in the record. A general statement ("flu is common in winter") \
or a question the writer asks is not a fact.
- Never invent a detail that the record does not state.
- If there are no facts, return {"facts": []}.

EXAMPLE 1
DOCUMENT:
Hello doctor, I am Rohan Mehta, a 41 year old bus driver from Hubli. For two months I have \
had chest tightness when I climb stairs, and my GP at Sparsh Clinic thinks it may be angina. \
I take aspirin. My wife Leela has psoriasis. Please call me on 98110 44521.
SUBJECTS:
- writer: the writer
- writer.wife: the writer's wife
OUTPUT:
{"facts": [
 {"subject": "writer", "relation": "identifier", "value": "Rohan Mehta", "importance": "low"},
 {"subject": "writer", "relation": "age", "value": "41 year old", "importance": "medium"},
 {"subject": "writer", "relation": "works_as", "value": "bus driver", "importance": "low"},
 {"subject": "writer", "relation": "lives_in", "value": "Hubli", "importance": "low"},
 {"subject": "writer", "relation": "has_symptom", "value": "chest tightness when I climb \
stairs", "importance": "high"},
 {"subject": "writer", "relation": "symptom_duration", "value": "two months", \
"importance": "medium"},
 {"subject": "writer", "relation": "treated_at", "value": "Sparsh Clinic", \
"importance": "medium"},
 {"subject": "writer", "relation": "has_condition", "value": "angina", "importance": "high"},
 {"subject": "writer", "relation": "took_medication", "value": "aspirin", "importance": "high"},
 {"subject": "writer", "relation": "contact", "value": "98110 44521", "importance": "low"},
 {"subject": "writer.wife", "relation": "relation_to", "value": "wife", "importance": "low"},
 {"subject": "writer.wife", "relation": "identifier", "value": "Leela", "importance": "low"},
 {"subject": "writer.wife", "relation": "has_condition", "value": "psoriasis", \
"importance": "high"}
]}

EXAMPLE 2
DOCUMENT:
Hi Dana, the vendor review has moved to 14 March. I will be on leave the week before for \
knee surgery, so please send me the slides early. Thanks, Victor Hale
SUBJECTS:
- writer: the writer
- person_1: Dana
OUTPUT:
{"facts": [
 {"subject": "writer", "relation": "identifier", "value": "Victor Hale", "importance": "low"},
 {"subject": "writer", "relation": "had_procedure", "value": "knee surgery", \
"importance": "high"},
 {"subject": "person_1", "relation": "identifier", "value": "Dana", "importance": "low"}
]}
(The vendor review date is about a meeting, not about a person, so it is not a fact.)
"""


def build_user_prompt(text: str, subjects: list[Subject], spans: list[SensitiveSpan]) -> str:
    people = "\n".join(f"- {s.id}: {s.label}" for s in subjects) or "- writer: the writer"
    hints = "\n".join(f"- {s.category.value}: {s.text}" for s in spans) or "(none)"
    return (
        f"DOCUMENT:\n{text}\n\nSUBJECTS:\n{people}\n\n"
        f"SENSITIVE DETAILS ALREADY DETECTED (each one should appear in a fact):\n{hints}\n\n"
        "List every fact in this document about the subjects above.\nOUTPUT:"
    )


async def extract_facts(
    text: str, subjects: list[Subject], spans: list[SensitiveSpan], llm: LLMClient
) -> list[FactTuple]:
    result = await llm.json_call(
        SYSTEM_PROMPT, build_user_prompt(text, subjects, spans), LLMFactList
    )
    return [
        FactTuple(
            subject=f.subject,
            relation=f.relation,
            value=f.value,
            importance=Importance(f.importance),
        )
        for f in result.facts
    ]
