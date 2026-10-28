"""Agent 3 - reconstruction prompt.  OWNER: Person C (Model, Platform and Integration).

``reconstruct`` writes a fresh document from the facts alone (docs/phase3_plan.md, Person C,
item 2). It never sees the original text, and it never sees the removed details either:
before anything reaches the model, every removed span in a fact value is rewritten with that
span's placeholder token or generalised phrase (``safe_value``), in any form the leak check
recognises (a name part, a part of a place, a reformatted number). ``safe_facts`` then drops
what must not reach the model at all: an identifier or contact detail that did not become a
token (Agent 1 missed it, or it is not an identifier: "the writer"), and a relationship given
to someone who is not one of the writer's relatives. The model's job is to turn the rest into
a short, natural record that a reader would accept as the original message with the
identifying details taken out.

``feedback`` (on the retry) lists what the leak check and the grounding check found in the
previous attempt; only the categories are shown to the model, never the leaked text.

The few-shot example is written for this prompt and must never be taken from
``data/fixtures`` or ``data/gold`` (``tests/prompts/test_prompts.py`` checks).
"""

from __future__ import annotations

import re

from sanitise.agents.rewriter import leakcheck, ontology
from sanitise.llm.base import LLMClient
from sanitise.schemas import Action, FactTuple, RoutingDecision, SensitiveSpan

TOKEN = re.compile(r"\[[A-Z]+(?:_[A-Z]+)*(?:_\d+)?\]")
IDENTIFYING = {"identifier", "contact"}

SYSTEM_PROMPT = """\
You rewrite a private record from a list of facts so it can be shared without identifying \
anyone. The facts have already been made safe: some details are placeholder tokens such as \
[PERSON_1] or [PHONE_1], and some are deliberately vague phrases such as "a town in Kerala".

Rules:
1. Use every fact, and only the facts. Never add a name, age, place, date, number, job, \
condition or any other detail that is not in the facts.
2. Copy placeholder tokens exactly, brackets included. Every token in the facts must appear \
in your record, contact details too.
3. Keep vague phrases vague. You may change their grammar to fit the sentence ("in their 40s" \
can become "in my 40s"), but never make them more specific. Fit them in naturally: "I am a \
working professional", not "I work as a working professional".
4. Write as the original writer would: first person for facts about "the writer" ("I", "my \
wife"). Two to five plain sentences, no headings, no lists. Say each detail once.
5. Leave out a fact that carries no information, such as a value that only says "the writer".
6. Return only the rewritten record.

EXAMPLE
FACTS:
the writer - name: [PERSON_1].
the writer - age: in their 40s.
the writer - works as: a transport worker.
the writer - lives in: a city in Karnataka.
the writer - has symptom: chest tightness when I climb stairs.
the writer - symptom duration: two months.
the writer - has condition: angina.
the writer - took medication: aspirin.
the writer - contact: [PHONE_1].
the writer's wife - relation to: wife.
the writer's wife - has condition: psoriasis.
OUTPUT:
I am [PERSON_1], in my 40s, and I work as a transport worker in a city in Karnataka. For two \
months I have had chest tightness when I climb stairs, and I have been told it may be angina. \
I take aspirin. My wife has psoriasis. You can reach me on [PHONE_1].
"""


def _outside_tokens(pattern: re.Pattern[str], decision: RoutingDecision, value: str) -> str:
    tokens = [(m.start(), m.end()) for m in TOKEN.finditer(value)]

    def sub(m: re.Match[str]) -> str:
        if any(a < m.end() and m.start() < b for a, b in tokens):
            return m.group(0)
        if decision.action is Action.GENERALISE:  # "Mulund" alone, not "a hospital"
            return ontology.generalise(m.group(0), decision.category)
        assert decision.replacement is not None
        return decision.replacement

    return pattern.sub(sub, value)


def safe_value(value: str, decisions: list[RoutingDecision]) -> str:
    """``value`` with every removed span replaced by its replacement: the exact text first
    (longest first), then any other form of it the leak check knows ("Daniel" from "Daniel
    Whitcomb" gets the full name's token; "Mulund" from "Fortis Hospital, Mulund" gets a
    generalised phrase of its own, so it does not become "a hospital")."""
    removed = sorted(
        (
            d
            for d in decisions
            if d.action is not Action.KEEP and d.replacement and d.span_text.strip()
        ),
        key=lambda d: len(d.span_text),
        reverse=True,
    )
    for d in removed:
        assert d.replacement is not None
        value = re.sub(re.escape(d.span_text.strip()), d.replacement, value, flags=re.IGNORECASE)
    for d in removed:
        assert d.replacement is not None
        for pattern in leakcheck.leak_patterns(d.span_text, d.category):
            value = _outside_tokens(pattern, d, value)
    return value


def safe_facts(
    facts: list[FactTuple], decisions: list[RoutingDecision]
) -> list[tuple[FactTuple, str]]:
    """The facts the model may see, each with its safe value (see the module docstring)."""
    out = []
    for f in facts:
        value = safe_value(f.value, decisions)
        if f.relation in IDENTIFYING and not TOKEN.search(value):
            continue
        if f.relation == "relation_to" and not f.subject.startswith("writer."):
            continue
        out.append((f, value))
    return out


def subject_label(subject: str) -> str:
    if subject == "writer":
        return "the writer"
    if subject.startswith("writer."):
        return "the writer's " + subject.removeprefix("writer.").replace("_", " ")
    return subject.replace("_", " ")


def relation_label(relation: str, value: str) -> str:
    """``identifier`` is shown as "name" or "ID number", so an ID is never written as a name
    ("I am [ID_1]")."""
    if relation == "identifier":
        return "name" if "[PERSON_" in value else "ID number"
    return relation.replace("_", " ")


def draft(facts: list[FactTuple], decisions: list[RoutingDecision]) -> str:
    """One line per fact the model may see, with removed details already replaced."""
    return "\n".join(
        f"{subject_label(f.subject)} - {relation_label(f.relation, value)}: {value}."
        for f, value in safe_facts(facts, decisions)
    )


def build_user_prompt(
    facts: list[FactTuple],
    decisions: list[RoutingDecision],
    feedback: list[SensitiveSpan] | None = None,
) -> str:
    prompt = f"FACTS:\n{draft(facts, decisions) or '(no facts)'}"
    if feedback:
        kinds = ", ".join(sorted({s.category.value for s in feedback}))
        prompt += (
            f"\n\nYour previous version contained details that must not appear, or that are "
            f"not in the facts ({kinds}). Write it again using only the tokens and phrases in "
            "the facts."
        )
    return prompt + "\nOUTPUT:"


async def reconstruct(
    facts: list[FactTuple],
    decisions: list[RoutingDecision],
    placeholders: dict[str, str],
    llm: LLMClient,
    feedback: list[SensitiveSpan] | None = None,
) -> str:
    return await llm.text_call(SYSTEM_PROMPT, build_user_prompt(facts, decisions, feedback))
