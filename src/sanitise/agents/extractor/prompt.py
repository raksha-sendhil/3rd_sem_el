"""Agent 2 - fact-extraction prompt.  OWNER: Person C (Model, Platform and Integration).

PLACEHOLDER: a minimal prompt with no few-shot examples, so the pipeline
runs end to end. Replace the prompt and add few-shot examples; keep the signature, and call
the model only through ``llm.json_call`` (docs/phase3_plan.md, Person C, item 2).

Inputs: the document, the people in it (``subjects``, from Person A - every fact's subject
must be one of their ids), and the detected sensitive spans (useful hints for what to
extract). Output: ``FactTuple``s with ``relation`` from ``schemas.RELATIONS``.

Decoding is constrained to ``LLMFactList`` below: ``relation`` and ``importance`` are enums,
so the model cannot invent a relation name. The pipeline (Person C) then runs Person A's
``normalise_facts`` and drops facts whose subject is unknown.
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
You extract the facts stated in a document as structured tuples.
Each fact is {"subject", "relation", "value", "importance"}:
- subject: one of the subject ids listed under SUBJECTS.
- relation: one of the allowed relation names.
- value: the fact's value in a few words, as stated in the document.
- importance: high for medical facts (conditions, symptoms, medication, procedures), \
medium for age, sex, place of treatment and dates, low for anything else.
Only state facts that the document says. Return {"facts": []} if there are none.
"""


def build_user_prompt(text: str, subjects: list[Subject], spans: list[SensitiveSpan]) -> str:
    people = "\n".join(f"- {s.id}: {s.label}" for s in subjects)
    hints = "\n".join(f"- {s.category.value}: {s.text}" for s in spans) or "(none)"
    return (
        f"DOCUMENT:\n{text}\n\nSUBJECTS:\n{people}\n\n"
        f"ALLOWED RELATIONS: {', '.join(RELATIONS)}\n\n"
        f"SENSITIVE DETAILS DETECTED (hints):\n{hints}\n\nOUTPUT:"
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
