"""Agent 3 - reconstruction prompt.  OWNER: Person C (Model, Platform and Integration).

``reconstruct`` writes a fresh document from the facts alone (docs/phase3_plan.md, Person C,
item 2). It never sees the original text, and it never sees the removed details either:
before anything reaches the model, every fact value that contains a removed span is rewritten
with that span's placeholder token or generalised phrase (``safe_value``). The model's job is
to turn those facts into a short, natural record that a reader would accept as the original
message with the identifying details taken out.

``feedback`` (on the retry) lists what the leak check found in the previous attempt; only the
categories are shown to the model, never the leaked text.

The few-shot example is written for this prompt and must never be taken from
``data/fixtures`` or ``data/gold`` (``tests/prompts/test_prompts.py`` checks).
"""

from __future__ import annotations

import re

from sanitise.domain import is_legal
from sanitise.llm.base import LLMClient
from sanitise.schemas import Action, FactTuple, RoutingDecision, SensitiveSpan

SYSTEM_PROMPT = """\
You rewrite a private record from a list of facts so it can be shared without identifying \
anyone. The facts have already been made safe: some details are placeholder tokens such as \
[PERSON_1] or [PHONE_1], and some are deliberately vague phrases such as "a town in Kerala".

Rules:
1. Use every fact, and only the facts. Never add a name, age, place, date, number, job, \
condition or any other detail that is not in the facts.
2. Copy placeholder tokens exactly, brackets included.
3. Keep vague phrases vague. You may change their grammar to fit the sentence ("in their 40s" \
can become "in my 40s"), but never make them more specific.
4. Write as the original writer would: first person for facts about "the writer" ("I", "my \
wife"). Two to five plain sentences, no headings, no lists.
5. Return only the rewritten record.

EXAMPLE
FACTS:
the writer - identifier: [PERSON_1].
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


LEGAL_SYSTEM_PROMPT = """\
You rewrite a document from a litigation firm's past matter from a list of facts, so it can \
be shared inside the firm without identifying the client or anyone else. The facts have \
already been made safe: some details are placeholder tokens such as [CLIENT_1], [PARTY_1] \
or [CASE_1], and some are deliberately vague phrases such as "a High Court" or "between \
₹1 lakh and ₹10 lakh".

Rules:
1. Use every fact, and only the facts. Never add a name, age, place, date, amount, court, \
case number or any other detail that is not in the facts.
2. Copy placeholder tokens exactly, brackets included.
3. Keep vague phrases vague; you may change their grammar to fit the sentence.
4. Keep the advice, the documents requested, the outcome and the next step in full: they \
are what the lawyers will search for.
5. Write a short case summary in plain sentences, three to six of them, no headings or lists.
6. Return only the rewritten document.

EXAMPLE
FACTS:
the client - identifier: [CLIENT_1].
the client - works as: a healthcare worker.
the client - lives in: a city in Karnataka.
the client - has legal issue: partition suit.
the client - opposing party: [PARTY_1].
the client - filed in: a civil court.
the client - advised: Ask the court to appoint a valuer before any division.
the client - outcome: The court appointed a valuer.
the client - next step: File objections to the valuation report.
OUTPUT:
[CLIENT_1], a healthcare worker in a city in Karnataka, brought a partition suit against \
[PARTY_1] before a civil court. We advised the client to ask the court to appoint a valuer \
before any division. The court appointed a valuer. The recommended next step is to file \
objections to the valuation report.
"""


def safe_value(value: str, decisions: list[RoutingDecision]) -> str:
    """``value`` with every removed span (longest first) replaced by its replacement."""
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
    return value


def subject_label(subject: str) -> str:
    if subject == "writer":
        return "the writer"
    if subject in {"client", "opposing_party", "witness"}:
        return "the " + subject.replace("_", " ")
    if subject.startswith("writer."):
        return "the writer's " + subject.removeprefix("writer.").replace("_", " ")
    return subject.replace("_", " ")


def draft(facts: list[FactTuple], decisions: list[RoutingDecision]) -> str:
    """One line per fact, with removed details already replaced."""
    return "\n".join(
        f"{subject_label(f.subject)} - {f.relation.replace('_', ' ')}: "
        f"{safe_value(f.value, decisions)}."
        for f in facts
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
            f"\n\nYour previous version contained details that must not appear ({kinds}). "
            "Write it again using only the tokens and phrases in the facts."
        )
    return prompt + "\nOUTPUT:"


async def reconstruct(
    facts: list[FactTuple],
    decisions: list[RoutingDecision],
    placeholders: dict[str, str],
    llm: LLMClient,
    feedback: list[SensitiveSpan] | None = None,
) -> str:
    system = LEGAL_SYSTEM_PROMPT if is_legal() else SYSTEM_PROMPT
    return await llm.text_call(system, build_user_prompt(facts, decisions, feedback))
