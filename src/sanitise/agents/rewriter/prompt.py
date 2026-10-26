"""Agent 3 - reconstruction prompt.  OWNER: Person C (Model, Platform and Integration).

PLACEHOLDER: a minimal prompt, so the pipeline runs end to end. Replace
the prompt and add few-shot examples; keep the signature, and call the model only through
``llm.text_call`` (docs/phase3_plan.md, Person C, item 2).

``reconstruct`` writes a fresh document from the facts alone. It never sees the original text,
and it never sees the removed details either: before anything reaches the model, every fact
value that contains a removed span is rewritten with that span's placeholder token or
generalised phrase (``_safe_value``). The model's job is to write natural, readable prose that
uses those tokens and phrases exactly as given.

``feedback`` (on the retry) lists what the leak check found in the previous attempt; only the
categories are shown to the model, never the leaked text.
"""

from __future__ import annotations

import re

from sanitise.llm.base import LLMClient
from sanitise.schemas import Action, FactTuple, RoutingDecision, SensitiveSpan

SYSTEM_PROMPT = """\
You rewrite a private record as a short, natural document written from the facts given.
Use placeholder tokens such as [PERSON_1] and the generalised phrases exactly as they appear \
in the facts. Do not add names, places, dates, numbers or any detail that is not in the facts.
Write in plain sentences. Return only the rewritten document.
"""


def _safe_value(value: str, decisions: list[RoutingDecision]) -> str:
    for d in decisions:
        if d.action is Action.KEEP or not d.replacement or not d.span_text.strip():
            continue
        value = re.sub(re.escape(d.span_text.strip()), d.replacement, value, flags=re.IGNORECASE)
    return value


def draft(facts: list[FactTuple], decisions: list[RoutingDecision]) -> str:
    """A plain sentence per fact, with removed details already replaced."""
    lines = []
    for f in facts:
        subject = f.subject.replace("writer.", "the writer's ").replace("writer", "the writer")
        lines.append(
            f"{subject} - {f.relation.replace('_', ' ')}: {_safe_value(f.value, decisions)}."
        )
    return "\n".join(lines)


def build_user_prompt(
    facts: list[FactTuple],
    decisions: list[RoutingDecision],
    feedback: list[SensitiveSpan] | None = None,
) -> str:
    prompt = f"FACTS:\n{draft(facts, decisions)}"
    if feedback:
        kinds = ", ".join(sorted({s.category.value for s in feedback}))
        prompt += (
            f"\n\nYour previous version still contained removed details ({kinds}). "
            "Use only the tokens and phrases given in the facts."
        )
    return prompt


async def reconstruct(
    facts: list[FactTuple],
    decisions: list[RoutingDecision],
    placeholders: dict[str, str],
    llm: LLMClient,
    feedback: list[SensitiveSpan] | None = None,
) -> str:
    return await llm.text_call(SYSTEM_PROMPT, build_user_prompt(facts, decisions, feedback))
