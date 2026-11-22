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

from sanitise_finance.agents.extractor import importance
from sanitise_finance.llm.base import LLMClient
from sanitise_finance.schemas import FactTuple, PIICategory, SensitiveSpan, Subject

C = PIICategory

# Built in code by the pipeline, never by the model.
IDENTITY_RELATIONS = ("identifier", "contact", "relation_to")
MODEL_RELATIONS = (
    "product",
    "issue",
    "amount",
    "account_event",
    "income",
    "employer",
    "age",
    "sex",
    "lives_in",
    "works_as",
    "studies_at",
    "has_condition",
    "event_date",
    "plan",
    "other",
)
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
    C.AMOUNT,
    C.PRODUCT,
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
You read one financial record (a customer's message, a form, a statement, a report or a \
workplace email) and list the facts it states about ONE subject, named under PERSON. "writer" \
is whoever wrote it, and also stands for the record itself: its amounts, products and figures \
belong to the writer unless another person is clearly their owner. Each fact is \
{"relation", "value"}.

relation: exactly one of these names:
- product: the financial product the record is about ("car loan", "savings account")
- issue: what went wrong, or what the record is about ("charged twice", "refund never came")
- amount: any figure, WITH the words that say what it is: a sum, fee, balance, limit, interest \
rate, percentage, term or count ("late fee: 25", "loan term: 36 months", "profit in 2019: 4,102")
- account_event: something that happened to an account, loan, card or payment (bounced, \
frozen, charged, declined, reversed, approved, closed)
- income: pay, salary, wages or other income, with its amount and period
- employer: the organisation someone works for
- age, sex: the person's age or sex
- lives_in: a town, city or address where the person lives
- works_as: the person's job or role; studies_at: a school or university they attend
- has_condition: an illness or disability the record mentions
- event_date: a date that matters to someone in the record
- plan: a request, a decision or a next step
- other: any other content that matters (a purpose, a reason, a condition of an agreement, a \
finding)

value: the detail in a few words, copied from the record, with every number exactly as written.

Rules:
- Every amount, product and issue in the record is a fact. Do not summarise or leave any out.
- Names, ID numbers, phone numbers, email addresses and family relationships are handled \
elsewhere: never list them.
- Only what the record states. If it says nothing about PERSON, return {"facts": []}.

EXAMPLE 1
DOCUMENT:
Hello, my car loan instalment of 8,250 was taken twice this month and the second debit \
bounced my rent cheque. Please reverse it. Ingrid Moe
PERSON: writer (the writer), called: I, my
OUTPUT:
{"facts": [
 {"relation": "product", "value": "car loan"},
 {"relation": "amount", "value": "car loan instalment of 8,250"},
 {"relation": "issue", "value": "instalment taken twice this month"},
 {"relation": "account_event", "value": "second debit bounced rent cheque"},
 {"relation": "plan", "value": "reverse it"}
]}
(The name is handled elsewhere.)
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
You read one financial record (a customer's message, a form, a statement, a report, an \
agreement or a workplace email) and list everything useful it states: its amounts, products, \
issues and figures, and what it says about the people in it. Each fact is {"subject", \
"relation", "value"}.

subject: an id from SUBJECTS. "writer" is whoever wrote the record, and also stands for the \
record itself: the figures of a report, the terms of an agreement and the lines of a statement \
have the subject "writer". "writer.<relative>" is one of their relatives; "person_<n>" is a \
person the record names. A fact about that person (their income, their employer, their job) \
has their id. A record that names no person still has facts.

relation: exactly one of these names:
- product: the financial product the record is about ("car loan", "savings account")
- issue: what went wrong, or what the record is about ("charged twice", "refund never came")
- amount: any figure, WITH the words that say what it is: a sum, fee, balance, limit, interest \
rate, percentage, term or count ("late fee: 25", "loan term: 36 months", "profit in 2019: 4,102")
- account_event: something that happened to an account, loan, card or payment (bounced, \
frozen, charged, declined, reversed, approved, closed)
- income: pay, salary, wages or other income, with its amount and period
- employer: the organisation someone works for
- age, sex: the person's age or sex
- lives_in: a town, city or address where the person lives
- works_as: the person's job or role; studies_at: a school or university they attend
- has_condition: an illness or disability the record mentions
- event_date: a date that matters to someone in the record
- plan: a request, a decision or a next step
- other: any other content that matters (a purpose, a reason, a condition of an agreement, a \
finding)

value: copied from the record's own wording, short, and complete on its own:
- Copy every number exactly as written, with its currency and unit. Never round, convert, add \
up or leave one out.
- Say what each number is: "opening balance: 7,410.20", not "7,410.20". In a table, put the \
row's label with the numbers, and the column's heading with each number. One row is one fact.
- Use the detail from the record, not a description of it ("purpose: new roof", not "gives a \
purpose").

Rules:
- List EVERY amount, rate, term, product, issue and event. A record full of numbers gives many \
facts. An empty blank or a placeholder such as "[Amount]" gives none.
- Names, ID numbers, account and card numbers, phone numbers, email addresses and family \
relationships are handled elsewhere: never list them, and leave them out of every value. \
Leave company names out of values too, except in employer.
- Only what the record states.
- Only if the record has no content at all, return {"facts": []}.

EXAMPLE 1
DOCUMENT:
Hello, I am Ingrid Moe. My car loan instalment of 8,250 was taken twice in May and the second \
debit bounced my rent cheque, so I was fined 400. I work at Fernhollow Dairy and earn 46,000 a \
month. Please reverse the debit.
SUBJECTS:
- writer: the writer (Ingrid Moe)
OUTPUT:
{"facts": [
 {"subject": "writer", "relation": "product", "value": "car loan"},
 {"subject": "writer", "relation": "amount", "value": "car loan instalment: 8,250"},
 {"subject": "writer", "relation": "issue", "value": "instalment taken twice in May"},
 {"subject": "writer", "relation": "account_event", "value": "second debit bounced rent cheque"},
 {"subject": "writer", "relation": "amount", "value": "fined 400 for the bounced cheque"},
 {"subject": "writer", "relation": "employer", "value": "Fernhollow Dairy"},
 {"subject": "writer", "relation": "income", "value": "earns 46,000 a month"},
 {"subject": "writer", "relation": "plan", "value": "reverse the debit"}
]}

EXAMPLE 2
DOCUMENT:
PERSONAL LOAN REQUEST
Applicant: Tobias Renn
Role: press operator
Salary: 2,900 a month
Amount requested: 12,500
Repay over: 48 months at 7.2%
Purpose: kitchen refit
SUBJECTS:
- writer: the writer
- person_1: Tobias Renn
OUTPUT:
{"facts": [
 {"subject": "writer", "relation": "product", "value": "personal loan"},
 {"subject": "person_1", "relation": "works_as", "value": "press operator"},
 {"subject": "person_1", "relation": "income", "value": "salary: 2,900 a month"},
 {"subject": "writer", "relation": "amount", "value": "amount requested: 12,500"},
 {"subject": "writer", "relation": "amount", "value": "repay over: 48 months"},
 {"subject": "writer", "relation": "amount", "value": "interest rate: 7.2%"},
 {"subject": "writer", "relation": "other", "value": "purpose: kitchen refit"}
]}

EXAMPLE 3
DOCUMENT:
Segment results (thousands)
                 2022     2021
Nurseries       6,614    6,020
Landscaping     1,187    1,342
Landscaping fell as two council contracts ended.
SUBJECTS:
- writer: the writer
OUTPUT:
{"facts": [
 {"subject": "writer", "relation": "amount", "value": "segment results, Nurseries (thousands): \
2022: 6,614; 2021: 6,020"},
 {"subject": "writer", "relation": "amount", "value": "segment results, Landscaping (thousands): \
2022: 1,187; 2021: 1,342"},
 {"subject": "writer", "relation": "other", "value": "Landscaping fell as two council contracts ended"}
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
        f"{hints}\n\nList every fact in this document, with every figure.\nOUTPUT:"
    )


async def extract_facts(
    text: str, subjects: list[Subject], spans: list[SensitiveSpan], llm: LLMClient
) -> list[FactTuple]:
    """One call per document for a strong model, one per subject otherwise. The importance of
    each fact is set here, by its relation."""
    subjects = subjects or [Subject(id="writer", label="the writer")]
    if getattr(llm, "one_call_per_document", False):
        result = await llm.json_call(
            DOCUMENT_PROMPT,
            build_document_prompt(text, subjects, spans),
            document_schema([s.id for s in subjects]),
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
