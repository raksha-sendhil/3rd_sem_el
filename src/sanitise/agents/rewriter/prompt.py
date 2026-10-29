"""Agent 3 - reconstruction prompt.  OWNER: Person C (Model, Platform and Integration).

``reconstruct`` writes a fresh document from the facts alone (docs/phase3_plan.md, Person C,
item 2). It never sees the original text, and it never sees the removed details either:
before anything reaches the model, every removed span in a fact value is rewritten with that
span's placeholder token or generalised phrase (``safe_value``), in any form the leak check
recognises (a name part, a part of a place, a reformatted number). ``safe_facts`` then drops
what must not reach the model at all: an identifier or contact detail that did not become a
token (Agent 1 missed it, or it is not an identifier: "the writer"), and every ``relation_to``
fact (the subject label already says "the writer's son"; a separate line made the model write
"my sister is my sister"). The model's job is to turn the rest into a short, natural record
that a reader would accept as the original message with the identifying details taken out.
A leading "Here is the rewritten record:" is removed from the reply.

``feedback`` (on the retry) lists what the leak check and the grounding check found in the
previous attempt; only the categories are shown to the model, never the leaked text.

The few-shot example is written for this prompt and must never be taken from
``data/fixtures`` or ``data/gold`` (``tests/prompts/test_prompts.py`` checks).
"""

from __future__ import annotations

import re

from sanitise.agents.rewriter import leakcheck, ontology
from sanitise.llm.base import LLMClient
from sanitise.schemas import Action, FactTuple, PIICategory, RoutingDecision, SensitiveSpan

TOKEN = re.compile(r"\[[A-Z]+(?:_[A-Z]+)*(?:_\d+)?\]")
IDENTIFYING = {"identifier", "contact"}
_WORD = re.compile(r"[A-Za-z0-9]+")
# "Here is the rewritten record:" and the like, before the record itself
PREAMBLE = re.compile(r"\A\s*(?:sure[,!.]?\s*)?(?:here\s+is|here's|below\s+is)[^:\n]*:\s*", re.I)

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
the writer's wife - has condition: psoriasis.
OUTPUT:
I am [PERSON_1], in my 40s, and I work as a transport worker in a city in Karnataka. For two \
months I have had chest tightness when I climb stairs, and I have been told it may be angina. \
I take aspirin. My wife has psoriasis. You can reach me on [PHONE_1].
"""


CASE_FILE_PROMPT = """\
You rewrite a clinical record (a case sheet, clinic note or discharge summary) from a list of \
facts so it can be shared without identifying anyone. The facts have already been made safe: \
some details are placeholder tokens such as [PERSON_1] or [ID_1], and some are deliberately \
vague phrases such as "a hospital" or "in their 50s".

Rules:
1. Use every fact, and only the facts. Never add a name, age, place, date, number, finding, \
result or any other detail that is not in the facts.
2. Copy placeholder tokens exactly, brackets included. Every token in the facts must appear \
in your record, contact details too.
3. Keep vague phrases vague; never make them more specific.
4. Write a short case summary in the third person ("The patient ..."), under these headings, \
each on its own line and only when there are facts for it: Patient, Complaints, History, \
Examination, Investigations, Treatment (procedures and medications given), Assessment \
(diagnoses and impressions), Plan. Plain sentences after each heading.
5. Keep every value, unit and word that says how something is going or how sure it is \
(improving, persistent, suspected, day 6) exactly as given. Keep the tense of the facts.
6. Keep clinical abbreviations as written and read them correctly; never turn a negative into \
a positive ("NKDA" is "no known drug allergies", not an allergy).
7. A vague date only says roughly when: write "in 2026", never "on 2026". Leave out a date \
whose event the facts do not say.
8. Return only the rewritten record.

EXAMPLE
FACTS:
the patient - name: [PERSON_1].
the patient - ID number: [ID_1].
the patient - age: in their 50s.
the patient - sex: M.
the patient - treated at: a hospital.
the patient - has condition: Right foot cellulitis.
the patient - has condition: Gout.
the patient - progress: Gout, uncontrolled.
the patient - has symptom: Swelling and redness of right foot.
the patient - symptom duration: 5 days.
the patient - progress: swelling reduced on IV antibiotics.
the patient - took medication: febuxostat 40 mg OD.
the patient - vital sign: BP 130/84.
the patient - test result: Uric acid 9.1 mg/dL.
the patient - allergy: penicillin (rash).
the patient - plan: Review in OPD after 1 week.
the patient's wife - contact: [PHONE_1].
OUTPUT:
Patient: [PERSON_1] ([ID_1]), male, in their 50s, treated at a hospital.
Complaints: Swelling and redness of the right foot for 5 days, reduced on IV antibiotics.
History: Gout, uncontrolled. Allergic to penicillin (rash).
Examination: BP 130/84.
Investigations: Uric acid 9.1 mg/dL.
Treatment: Febuxostat 40 mg OD.
Assessment: Right foot cellulitis.
Plan: Review in OPD after 1 week. The patient's wife can be reached on [PHONE_1].
"""


def is_case_file(facts: list[FactTuple]) -> bool:
    """Agent 2 names a clinical record's people ``patient`` and ``patient.<kin>``."""
    return any(f.subject == "patient" or f.subject.startswith("patient.") for f in facts)


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
        # whole words only: "M" (a sex marker) must not hit the m of "antimalarials", nor the
        # age "9" the 9 of "Hb 9.8"
        span = rf"(?<!\w)(?<!\d\.){re.escape(d.span_text.strip())}(?!\w)(?!\.\d)"
        if d.action is Action.GENERALISE and re.search(span, value, re.IGNORECASE):
            # a vague phrase cannot replace a few words inside a longer one ("tore my anterior
            # cruciate ligament" -> "tore my a joint injury"); when the detail is most of the
            # value, the vague phrase is the whole value
            if 2 * len(_WORD.findall(d.span_text)) >= len(_WORD.findall(TOKEN.sub(" ", value))):
                value = d.replacement
                continue
        value = re.sub(span, d.replacement, value, flags=re.IGNORECASE)
    for d in removed:
        assert d.replacement is not None
        for pattern in leakcheck.leak_patterns(d.span_text, d.category):
            value = _outside_tokens(pattern, d, value)
    for d in removed:
        # last resort: a distinctive word of a removed name or place still in the value
        # ("a hospital, Kengeri" when "Blossom Hospital, Kengeri branch" was removed) means the
        # value is that detail in another form, so the whole value becomes its replacement
        if d.category in _NAMED and _leftover(value, d.span_text):
            assert d.replacement is not None
            value = d.replacement
    # a date made vague to its year: "Review on 23/10/2026" -> "Review in 2026"
    return _ON_YEAR.sub(lambda m: ("In " if m.group(1) == "O" else "in ") + m.group(2), value)


_ON_YEAR = re.compile(r"\b([Oo])n (\d{4}|the \d{4}s)\b")


_NAMED = {PIICategory.NAME, PIICategory.LOCATION, PIICategory.ADDRESS}
_GENERIC_PLACE_WORDS = {
    "hospital", "hospitals", "clinic", "clinics", "medical", "centre", "center", "health",
    "road", "street", "lane", "avenue", "nagar", "branch", "school", "college", "university",
    "apartments", "apartment", "flat", "house", "tower", "towers", "court", "block", "sector",
    "phase", "main", "cross", "north", "south", "east", "west", "city", "town", "district",
    "state", "the",
}  # fmt: skip


def _leftover(value: str, span_text: str) -> bool:
    words = {w for w in re.findall(r"[A-Z][A-Za-z'-]{3,}", span_text)} - {
        w.title() for w in _GENERIC_PLACE_WORDS
    }
    outside = TOKEN.sub(" ", value)
    return any(re.search(rf"(?<![A-Za-z]){re.escape(w)}(?![A-Za-z])", outside) for w in words)


def safe_facts(
    facts: list[FactTuple], decisions: list[RoutingDecision]
) -> list[tuple[FactTuple, str]]:
    """The facts the model may see, each with its safe value (see the module docstring)."""
    out = []
    for f in facts:
        value = safe_value(f.value, decisions)
        if f.relation in IDENTIFYING and TOKEN.sub("", value).strip(" ,;/"):
            continue  # not exactly a token: missed, garbled ("[ID_1]25") or not an identifier
        if f.relation == "relation_to":  # the label says it: "the writer's son"
            continue
        out.append((f, value))
    return out


_OTHERS = ["another person", "a second other person", "a third other person"]


def subject_labels(safe: list[tuple[FactTuple, str]]) -> dict[str, str]:
    """How each subject is named to the model. The writer and their relatives by role ("the
    writer's son"); anyone else by their name token ("[PERSON_2]") or, with none, as "another
    person", never by an id such as "person 3", which the model would copy into the record."""
    labels: dict[str, str] = {}
    unnamed = 0
    for f, _ in safe:
        subject = f.subject
        if subject in labels:
            continue
        if subject == "patient":
            labels[subject] = "the patient"
        elif subject.startswith("patient."):
            labels[subject] = "the patient's " + subject.removeprefix("patient.").replace("_", " ")
        elif subject == "writer":
            labels[subject] = "the writer"
        elif subject.startswith("writer."):
            labels[subject] = "the writer's " + subject.removeprefix("writer.").replace("_", " ")
        else:
            name = next(
                (
                    v
                    for g, v in safe
                    if g.subject == subject and g.relation == "identifier" and "[PERSON_" in v
                ),
                None,
            )
            if name is None:
                name = _OTHERS[min(unnamed, len(_OTHERS) - 1)]
                unnamed += 1
            labels[subject] = name
    return labels


def relation_label(relation: str, value: str) -> str:
    """``identifier`` is shown as "name" or "ID number", so an ID is never written as a name
    ("I am [ID_1]")."""
    if relation == "identifier":
        return "name" if "[PERSON_" in value else "ID number"
    return relation.replace("_", " ")


def draft(facts: list[FactTuple], decisions: list[RoutingDecision]) -> str:
    """One line per fact the model may see, with removed details already replaced. A line is
    never repeated ("Laura Fenn" and "Laura" are one token, so one name line), and a person
    named by their token gets no separate line saying their name is that token."""
    safe = safe_facts(facts, decisions)
    labels = subject_labels(safe)
    lines: list[str] = []
    for f, value in safe:
        label = labels[f.subject]
        if f.relation == "identifier" and value == label:
            continue
        line = f"{label} - {relation_label(f.relation, value)}: {value}."
        if line not in lines:
            lines.append(line)
    return "\n".join(lines)


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
    system = CASE_FILE_PROMPT if is_case_file(facts) else SYSTEM_PROMPT
    reply = await llm.text_call(system, build_user_prompt(facts, decisions, feedback))
    return PREAMBLE.sub("", reply).strip()
