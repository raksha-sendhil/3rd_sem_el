"""Agent 2 - fact-extraction prompt.  OWNER: Person C (Model, Platform and Integration).

Two ways to ask, chosen by the client (``one_call_per_document``):

* a strong model (Gemini) gets **one call per document** (``DOCUMENT_PROMPT``): the whole
  record, its people and Agent 1's details, and returns every person's facts. The subject of
  each fact must be one of the document's subject ids: the schema built per document
  (``document_schema``) makes them an enum.
* a small local model gets **one call per person** (``SYSTEM_PROMPT``), described below.

One small constrained-JSON call per person in the document (docs/phase3_plan.md, Person C,
item 2). Each call gets the document, the one person to describe (an id from Person A's
``subjects``, a label and the words the text uses for them) and the details Agent 1 found, and
returns that person's facts. A small model follows one narrow question ("what does the record
say about the writer's son?") far more reliably than one call that has to find every person,
assign every fact and pick from every relation at once.

Decoding is constrained to ``LLMFactList``: ``relation`` is an enum, so the model cannot
invent a relation name. The model never decides a fact's subject (the call does) or its
importance (``importance.default_importance``, a fixed rule per relation).

Names, ID numbers, contact details and kinship are not asked for at all: the extractor
pipeline builds those facts from Agent 1's spans and Person A's subjects, in code
(``pipeline.identity_facts``). The model cannot mistake "the writer" for a name, copy an ID
number wrongly or invent a relative.

The few-shot examples are written for this prompt. They must never be taken from
``data/fixtures`` or ``data/gold``, because the evaluation runs on those documents
(``tests/prompts/test_prompts.py`` fails if a gold phrase appears here).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from sanitise.agents.extractor import importance
from sanitise.casefile import is_case_file
from sanitise.llm.base import LLMClient
from sanitise.schemas import FactTuple, PIICategory, SensitiveSpan, Subject

C = PIICategory

# Built in code by the pipeline, never by the model.
IDENTITY_RELATIONS = ("identifier", "contact", "relation_to")
MODEL_RELATIONS = (
    "age",
    "sex",
    "lives_in",
    "works_as",
    "studies_at",
    "treated_at",
    "has_condition",
    "has_symptom",
    "symptom_duration",
    "took_medication",
    "had_procedure",
    "event_date",
    "height",
    "weight",
    "other",
)
# A clinical record's further relations (``CASE_DOCUMENT_PROMPT``); other documents use only
# MODEL_RELATIONS.
CASE_RELATIONS = (*MODEL_RELATIONS, "progress", "vital_sign", "test_result", "allergy",
                  "assessment", "plan")  # fmt: skip
# Agent 1's hints for the categories the model extracts; identifiers are left out.
HINT_CATEGORIES = {
    C.AGE,
    C.DATE,
    C.ADDRESS,
    C.LOCATION,
    C.OCCUPATION,
    C.PHYSICAL,
    C.MEDICAL_CONDITION,
    C.SYMPTOM,
}

Relation = Literal[MODEL_RELATIONS]  # type: ignore[valid-type]


class LLMFact(BaseModel):
    relation: Relation
    value: str = Field(min_length=1, max_length=300)


class LLMFactList(BaseModel):
    """The schema handed to Ollama. Titled "FactList" (MockLLM scripts use that name)."""

    model_config = ConfigDict(title="FactList")
    facts: list[LLMFact]


SYSTEM_PROMPT = """\
You read one private record (a patient's message to a doctor, or a workplace email) and list \
the facts it states about ONE person, named under PERSON. Each fact is {"relation", "value"}.

relation: exactly one of these names:
- age, sex: the person's age or sex
- lives_in: a town, city or address where the person lives
- works_as: the person's job or role; studies_at: a school or university they attend
- treated_at: a named hospital, clinic or doctor's practice treating them
- has_condition: a named diagnosis, disease or clinical finding (also a suspected one)
- has_symptom: something the person feels or shows (fever, cough, pain somewhere, swelling); \
not what triggers it
- symptom_duration: how long a symptom has lasted ("three weeks")
- took_medication, had_procedure: a drug they take; an operation or test they had
- event_date: a specific date of an event in their life (diagnosis, admission, leave)
- height, weight: body measurements
- other: anything else the record says that describes the person

value: the detail in a few words, copied from the record's own wording.

Rules:
- Only facts about PERSON. A fact about anyone else belongs to them, not to PERSON.
- Names, ID numbers, phone numbers, email addresses and family relationships are handled \
elsewhere: never list them.
- Only what the record states. Attending a school is studies_at, not works_as "student"; \
symptom_duration is a length of time, not an activity.
- A general statement ("flu is common in winter") or a question is not a fact.
- If the record says nothing about PERSON, return {"facts": []}.

EXAMPLE 1
DOCUMENT:
Hello doctor, I am Rohan Mehta, a 41 year old bus driver from Hubli. For two months I have \
had chest tightness when I climb stairs, and my GP at Sparsh Clinic thinks it may be angina. \
I take aspirin. My wife Leela has psoriasis. Please call me on 98110 44521.
PERSON: writer (the writer), called: I, my, me
OUTPUT:
{"facts": [
 {"relation": "age", "value": "41 year old"},
 {"relation": "works_as", "value": "bus driver"},
 {"relation": "lives_in", "value": "Hubli"},
 {"relation": "has_symptom", "value": "chest tightness when I climb stairs"},
 {"relation": "symptom_duration", "value": "two months"},
 {"relation": "treated_at", "value": "Sparsh Clinic"},
 {"relation": "has_condition", "value": "angina"},
 {"relation": "took_medication", "value": "aspirin"}
]}
(The name and the phone number are handled elsewhere. The psoriasis is the wife's.)

EXAMPLE 2
DOCUMENT: (the same record)
PERSON: writer.wife (the writer's wife), called: My wife, Leela
OUTPUT:
{"facts": [
 {"relation": "has_condition", "value": "psoriasis"}
]}

EXAMPLE 3
DOCUMENT:
Good evening doctor. My son is 9 and goes to Riverside Public School in Mysuru. Since 2 May \
he has had itchy eyes and sneezing every morning. His hospital number is MR-55821. You can \
email me at k.rao.home@mailbox.test.
PERSON: writer.son (the writer's son), called: My son, he, His
OUTPUT:
{"facts": [
 {"relation": "age", "value": "9"},
 {"relation": "studies_at", "value": "Riverside Public School"},
 {"relation": "lives_in", "value": "Mysuru"},
 {"relation": "has_symptom", "value": "itchy eyes"},
 {"relation": "has_symptom", "value": "sneezing every morning"},
 {"relation": "event_date", "value": "2 May"}
]}

EXAMPLE 4
DOCUMENT:
Hi Dana, the vendor review has moved to 14 March. I will be on leave the week before for \
knee surgery, so please send me the slides early. Thanks, Victor Hale
PERSON: person_1 (Dana), called: Dana
OUTPUT:
{"facts": []}
(The record asks Dana for slides but states nothing about her. The review date is about a \
meeting, not a person.)
"""


def build_user_prompt(text: str, subject: Subject, spans: list[SensitiveSpan]) -> str:
    called = ", ".join(subject.mentions) or subject.label
    hints = (
        "\n".join(f"- {s.category.value}: {s.text}" for s in spans if s.category in HINT_CATEGORIES)
        or "(none)"
    )
    return (
        f"DOCUMENT:\n{text}\n\n"
        f"DETAILS FOUND IN THE RECORD (about anyone; use those about PERSON):\n{hints}\n\n"
        f"PERSON: {subject.id} ({subject.label}), called: {called}\n"
        f"List the facts the record states about {subject.label}.\nOUTPUT:"
    )


DOCUMENT_PROMPT = """\
You read one private record (a patient's message to a doctor, or a workplace email) and list \
the facts it states about the people in it. Each fact is {"subject", "relation", "value"}.

subject: the id of the person the fact is about, from SUBJECTS. "writer" is the person who \
wrote the record; "writer.<relative>" is one of their relatives; "person_<n>" is anyone else, \
such as a doctor or a colleague. Work out who each sentence is about: "he", "she" and "his" \
refer to the person the sentence is about, which is not always the person named last. A \
doctor who treats someone is not the patient: the patient's symptoms, conditions and \
treatment belong to the patient.

relation: exactly one of these names:
- age, sex: the person's age or sex
- lives_in: a town, city or address where the person lives
- works_as: the person's job or role; studies_at: a school or university they attend
- treated_at: a hospital, clinic or doctor's practice where they are treated or seen
- has_condition: a named diagnosis, disease or clinical finding (also a suspected one)
- has_symptom: something the person feels or shows (fever, cough, pain somewhere); not what \
triggers it
- symptom_duration: how long a symptom has lasted ("three weeks", "over the last week")
- took_medication, had_procedure: a drug they take; an operation or test they had
- event_date: a specific date of an event in their life (diagnosis, admission, leave)
- height, weight: body measurements
- other: anything else the record states that matters about the person's health or \
situation, such as smoking history ("former smoker")

value: the detail in a few words, copied from the record's own wording.

Rules:
- Names, ID numbers, phone numbers, email addresses and family relationships are handled \
elsewhere: never list them.
- Only what the record states, about the people in it. Attending a school is studies_at, not \
works_as "student". A habit or a past habit is not a job. A general statement or a question \
("is a cough for months normal?") is not a fact, but the details it gives about the writer \
are ("cough", "over 6 months").
- One fact per detail; never list the same detail for two people.
- If there are no facts, return {"facts": []}.

EXAMPLE 1
DOCUMENT:
Hello doctor, I am Rohan Mehta, a 41 year old bus driver from Hubli. For two months I have \
had chest tightness when I climb stairs, and my GP at Sparsh Clinic thinks it may be angina. \
I take aspirin. My wife Leela has psoriasis. Please call me on 98110 44521.
SUBJECTS:
- writer: the writer (Rohan Mehta)
- writer.wife: the writer's wife (Leela)
OUTPUT:
{"facts": [
 {"subject": "writer", "relation": "age", "value": "41 year old"},
 {"subject": "writer", "relation": "works_as", "value": "bus driver"},
 {"subject": "writer", "relation": "lives_in", "value": "Hubli"},
 {"subject": "writer", "relation": "has_symptom", "value": "chest tightness when I climb stairs"},
 {"subject": "writer", "relation": "symptom_duration", "value": "two months"},
 {"subject": "writer", "relation": "treated_at", "value": "Sparsh Clinic"},
 {"subject": "writer", "relation": "has_condition", "value": "angina"},
 {"subject": "writer", "relation": "took_medication", "value": "aspirin"},
 {"subject": "writer.wife", "relation": "has_condition", "value": "psoriasis"}
]}

EXAMPLE 2
DOCUMENT:
Good evening doctor. My son is 9 and goes to Riverside Public School in Mysuru. Since 2 May \
he has had itchy eyes and sneezing every morning. His hospital number is MR-55821. You can \
email me at k.rao.home@mailbox.test.
SUBJECTS:
- writer: the writer
- writer.son: the writer's son
OUTPUT:
{"facts": [
 {"subject": "writer.son", "relation": "age", "value": "9"},
 {"subject": "writer.son", "relation": "studies_at", "value": "Riverside Public School"},
 {"subject": "writer.son", "relation": "lives_in", "value": "Mysuru"},
 {"subject": "writer.son", "relation": "has_symptom", "value": "itchy eyes"},
 {"subject": "writer.son", "relation": "has_symptom", "value": "sneezing every morning"},
 {"subject": "writer.son", "relation": "event_date", "value": "2 May"}
]}

EXAMPLE 3
DOCUMENT:
Hi Dana, the vendor review has moved to 14 March. I will be on leave the week before for \
knee surgery, so please send me the slides early. Thanks, Victor Hale
SUBJECTS:
- writer: the writer (Victor Hale)
- person_1: Dana
OUTPUT:
{"facts": [
 {"subject": "writer", "relation": "had_procedure", "value": "knee surgery"}
]}
(The review date is about a meeting, not a person; the record states nothing about Dana.)
"""

# Clinical records only (``sanitise.casefile``): the six clinical relations, the words that
# say how something is going, the patient as the subject, and a discharge-summary example.
CASE_DOCUMENT_PROMPT = """\
You read one private record (a patient's message to a doctor, a workplace email or a clinical \
record) and list the facts it states about the people in it. Each fact is {"subject", \
"relation", "value"}.

subject: the id of the person the fact is about, from SUBJECTS. "writer" is the person who \
wrote the record; "writer.<relative>" is one of their relatives; "person_<n>" is anyone else, \
such as a doctor or a colleague. Work out who each sentence is about: "he", "she" and "his" \
refer to the person the sentence is about, which is not always the person named last. A \
doctor who treats someone is not the patient: the patient's symptoms, conditions and \
treatment belong to the patient.

relation: exactly one of these names:
- age, sex: the person's age or sex
- lives_in: a town, city or address where the person lives
- works_as: the person's job or role; studies_at: a school or university they attend
- treated_at: a hospital, clinic or doctor's practice where they are treated or seen
- has_condition: a named diagnosis, disease or clinical finding (also a suspected one)
- has_symptom: something the person feels or shows (fever, cough, pain somewhere); not what \
triggers it
- symptom_duration: how long a symptom has lasted ("three weeks", "over the last week")
- took_medication, had_procedure: a drug they take; an operation or test they had
- event_date: a specific date of an event in their life (diagnosis, admission, leave)
- height, weight: body measurements
- progress: how a symptom, condition or treatment is going, as the record says ("cough now \
improving", "headaches remain", "fever persistent on antimalarials", "day 6 of chloroquine")
- vital_sign: a measured vital sign with its value ("Temp 102.4 F", "BP 130/84")
- test_result: a test, scan or lab result with its value or finding ("Hb 9.8 g/dL", "smear \
positive for P. vivax", "X-ray: no fracture")
- allergy: an allergy, or that there is none ("penicillin (rash)", "no known drug allergies")
- assessment: the clinician's impression or a suspected cause ("?inadequate response")
- plan: a next step: a test ordered, advice, a review ("repeat CBC in 48 h")
- other: anything else the record states that matters about the person's health or \
situation, such as smoking history ("former smoker") or a finding that is absent ("no \
icterus")

value: the detail in a few words, copied from the record's own wording. Keep the words that \
say how something is going or how sure it is (improving, still, persistent, resolved, \
suspected): "My cough has got better but the headaches remain" gives has_symptom "cough", \
progress "cough has got better", has_symptom "headaches", progress "headaches remain".

Clinical records (case sheets, clinic notes, discharge summaries) are written by a clinician \
about a patient: their facts belong to "the patient" in SUBJECTS, and a relative listed there \
is the patient's. Read the abbreviations (C/O complaints, H/O history, K/c/o known case of, \
O/E on examination, Inv investigations, Imp impression, NKDA no known drug allergies, SOS when \
needed) and list every complaint, finding, result, medication, assessment and plan.

Rules:
- Names, ID numbers, phone numbers, email addresses and family relationships are handled \
elsewhere: never list them.
- Only what the record states, about the people in it. Attending a school is studies_at, not \
works_as "student". A habit or a past habit is not a job. A general statement or a question \
("is a cough for months normal?") is not a fact, but the details it gives about the writer \
are ("cough", "over 6 months").
- One fact per detail; never list the same detail for two people.
- If there are no facts, return {"facts": []}.

EXAMPLE 1
DOCUMENT:
Hello doctor, I am Rohan Mehta, a 41 year old bus driver from Hubli. For two months I have \
had chest tightness when I climb stairs, and my GP at Sparsh Clinic thinks it may be angina. \
I take aspirin. My wife Leela has psoriasis. Please call me on 98110 44521.
SUBJECTS:
- writer: the writer (Rohan Mehta)
- writer.wife: the writer's wife (Leela)
OUTPUT:
{"facts": [
 {"subject": "writer", "relation": "age", "value": "41 year old"},
 {"subject": "writer", "relation": "works_as", "value": "bus driver"},
 {"subject": "writer", "relation": "lives_in", "value": "Hubli"},
 {"subject": "writer", "relation": "has_symptom", "value": "chest tightness when I climb stairs"},
 {"subject": "writer", "relation": "symptom_duration", "value": "two months"},
 {"subject": "writer", "relation": "treated_at", "value": "Sparsh Clinic"},
 {"subject": "writer", "relation": "has_condition", "value": "angina"},
 {"subject": "writer", "relation": "took_medication", "value": "aspirin"},
 {"subject": "writer.wife", "relation": "has_condition", "value": "psoriasis"}
]}

EXAMPLE 2
DOCUMENT:
Good evening doctor. My son is 9 and goes to Riverside Public School in Mysuru. Since 2 May \
he has had itchy eyes and sneezing every morning. His hospital number is MR-55821. You can \
email me at k.rao.home@mailbox.test.
SUBJECTS:
- writer: the writer
- writer.son: the writer's son
OUTPUT:
{"facts": [
 {"subject": "writer.son", "relation": "age", "value": "9"},
 {"subject": "writer.son", "relation": "studies_at", "value": "Riverside Public School"},
 {"subject": "writer.son", "relation": "lives_in", "value": "Mysuru"},
 {"subject": "writer.son", "relation": "has_symptom", "value": "itchy eyes"},
 {"subject": "writer.son", "relation": "has_symptom", "value": "sneezing every morning"},
 {"subject": "writer.son", "relation": "event_date", "value": "2 May"}
]}

EXAMPLE 3
DOCUMENT:
Hi Dana, the vendor review has moved to 14 March. I will be on leave the week before for \
knee surgery, so please send me the slides early. Thanks, Victor Hale
SUBJECTS:
- writer: the writer (Victor Hale)
- person_1: Dana
OUTPUT:
{"facts": [
 {"subject": "writer", "relation": "had_procedure", "value": "knee surgery"}
]}
(The review date is about a meeting, not a person; the record states nothing about Dana.)

EXAMPLE 4
DOCUMENT:
DISCHARGE SUMMARY - Lakeview General Hospital, Nashik
Name: Prakash Deshmukh    Age/Sex: 58/M    IP No: LGH/24/1180
Admitted: 02/02/2024    Discharged: 06/02/2024
Diagnosis: Right foot cellulitis. Gout, uncontrolled.
Course: Swelling and redness of right foot x 5 days, reduced on IV antibiotics. Fever \
settled by day 3. No h/o trauma.
Vitals at discharge: BP 130/84, pulse 82/min.
Inv: Uric acid 9.1 mg/dL. WBC 13,600, down to 9,200. Allergy: penicillin (rash).
Advice: Tab. febuxostat 40 mg OD. Dressing on alternate days. Review in OPD after 1 week. \
Wife Sunita to check his diet.
SUBJECTS:
- writer: the clinician who wrote the record
- person_1: the patient (Prakash Deshmukh)
- writer.wife: the patient's wife (Sunita)
OUTPUT:
{"facts": [
 {"subject": "person_1", "relation": "age", "value": "58"},
 {"subject": "person_1", "relation": "sex", "value": "M"},
 {"subject": "person_1", "relation": "treated_at", "value": "Lakeview General Hospital"},
 {"subject": "person_1", "relation": "event_date", "value": "Admitted: 02/02/2024"},
 {"subject": "person_1", "relation": "event_date", "value": "Discharged: 06/02/2024"},
 {"subject": "person_1", "relation": "has_condition", "value": "Right foot cellulitis"},
 {"subject": "person_1", "relation": "has_condition", "value": "Gout"},
 {"subject": "person_1", "relation": "progress", "value": "Gout, uncontrolled"},
 {"subject": "person_1", "relation": "has_symptom", "value": "Swelling and redness of right foot"},
 {"subject": "person_1", "relation": "symptom_duration", "value": "5 days"},
 {"subject": "person_1", "relation": "progress", "value": "swelling reduced on IV antibiotics"},
 {"subject": "person_1", "relation": "has_symptom", "value": "Fever"},
 {"subject": "person_1", "relation": "progress", "value": "Fever settled by day 3"},
 {"subject": "person_1", "relation": "other", "value": "No h/o trauma"},
 {"subject": "person_1", "relation": "took_medication", "value": "IV antibiotics"},
 {"subject": "person_1", "relation": "vital_sign", "value": "BP 130/84"},
 {"subject": "person_1", "relation": "vital_sign", "value": "pulse 82/min"},
 {"subject": "person_1", "relation": "test_result", "value": "Uric acid 9.1 mg/dL"},
 {"subject": "person_1", "relation": "test_result", "value": "WBC 13,600, down to 9,200"},
 {"subject": "person_1", "relation": "allergy", "value": "penicillin (rash)"},
 {"subject": "person_1", "relation": "took_medication", "value": "febuxostat 40 mg OD"},
 {"subject": "person_1", "relation": "plan", "value": "Dressing on alternate days"},
 {"subject": "person_1", "relation": "plan", "value": "Review in OPD after 1 week"},
 {"subject": "writer.wife", "relation": "other", "value": "to check his diet"}
]}
"""

_PRONOUNS = {
    "i", "me", "my", "mine", "we", "us", "our", "he", "him", "his", "she", "her", "hers", "they",
    "them", "their",
}  # fmt: skip


def _names(subject: Subject) -> str:
    """The names a subject goes by, without pronouns: "the writer (Rohan Mehta)"."""
    names = [m for m in subject.mentions if m.lower() not in _PRONOUNS and m != subject.label]
    return f"{subject.label} ({', '.join(names)})" if names else subject.label


def document_schema(
    subject_ids: list[str], relations: tuple[str, ...] = MODEL_RELATIONS
) -> type[BaseModel]:
    """``FactList`` whose ``subject`` must be one of this document's subject ids."""
    fact = create_model(
        "LLMDocumentFact",
        subject=(Literal[tuple(subject_ids)], ...),  # type: ignore[valid-type]
        relation=(Literal[relations], ...),  # type: ignore[valid-type]
        value=(str, Field(min_length=1, max_length=300)),
    )
    return create_model(
        "LLMDocumentFactList", __config__=ConfigDict(title="FactList"), facts=(list[fact], ...)
    )


def build_document_prompt(text: str, subjects: list[Subject], spans: list[SensitiveSpan]) -> str:
    people = "\n".join(f"- {s.id}: {_names(s)}" for s in subjects)
    hints = (
        "\n".join(f"- {s.category.value}: {s.text}" for s in spans if s.category in HINT_CATEGORIES)
        or "(none)"
    )
    return (
        f"DOCUMENT:\n{text}\n\nSUBJECTS:\n{people}\n\n"
        "DETAILS FOUND IN THE RECORD (each belongs to someone; some may be mislabelled):\n"
        f"{hints}\n\nList every fact in this document about the subjects above.\nOUTPUT:"
    )


async def extract_facts(
    text: str, subjects: list[Subject], spans: list[SensitiveSpan], llm: LLMClient
) -> list[FactTuple]:
    """One call per document for a strong model, one per subject otherwise. The importance of
    each fact is set here, by its relation."""
    subjects = subjects or [Subject(id="writer", label="the writer")]
    if getattr(llm, "one_call_per_document", False):
        case = is_case_file(text)
        result = await llm.json_call(
            CASE_DOCUMENT_PROMPT if case else DOCUMENT_PROMPT,
            build_document_prompt(text, subjects, spans),
            document_schema([s.id for s in subjects], CASE_RELATIONS if case else MODEL_RELATIONS),
        )
        return [
            FactTuple(
                subject=f.subject,
                relation=f.relation,
                value=f.value,
                importance=importance.default_importance(f.relation),
            )
            for f in result.facts  # type: ignore[attr-defined]
        ]
    facts = []
    for subject in subjects:
        result = await llm.json_call(
            SYSTEM_PROMPT, build_user_prompt(text, subject, spans), LLMFactList
        )
        facts += [
            FactTuple(
                subject=subject.id,
                relation=f.relation,
                value=f.value,
                importance=importance.default_importance(f.relation),
            )
            for f in result.facts
        ]
    return facts
