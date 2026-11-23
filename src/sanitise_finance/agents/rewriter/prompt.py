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

``missing_figures`` is the backstop for "amounts must survive": a figure of the safe facts
that the model left out is added back, from the safe fact, after the checks.

``feedback`` (on the retry) lists what the leak check and the grounding check found in the
previous attempt; only the categories are shown to the model, never the leaked text.

The few-shot example is written for this prompt and must never be taken from
``data/fixtures`` or ``data/gold`` (``tests/prompts/test_prompts.py`` checks).
"""

from __future__ import annotations

import re
from contextvars import ContextVar

from sanitise_finance.agents.rewriter import fidelity, leakcheck, ontology
from sanitise_finance.casefile import CASE_FILE
from sanitise_finance.llm.base import LLMClient
from sanitise_finance.schemas import Action, FactTuple, PIICategory, RoutingDecision, SensitiveSpan

TOKEN = re.compile(r"\[[A-Z]+(?:_[A-Z]+)*(?:_\d+)?\]")
IDENTIFYING = {"identifier", "contact"}
_WORD = re.compile(r"[A-Za-z0-9]+")
# "Here is the rewritten record:" and the like, before the record itself
PREAMBLE = re.compile(r"\A\s*(?:sure[,!.]?\s*)?(?:here\s+is|here's|below\s+is)[^:\n]*:\s*", re.I)

SYSTEM_PROMPT = """\
You rewrite a private financial record from a list of facts so it can be shared without \
identifying anyone. The facts have already been made safe: some details are placeholder tokens \
such as [PERSON_1] or [ID_1], and some are deliberately vague phrases such as "a public-sector \
bank" or "a company".

Rules:
1. Use every fact, and only the facts. Never add a name, age, place, date, number, product or \
any other detail that is not in the facts.
2. Copy placeholder tokens exactly, brackets included. Every token in the facts must appear \
in your record.
3. Copy every amount, rate, percentage, term and year exactly as the facts write it, next to \
the words that say what it is. Never round, convert, total or drop one, however many there are.
4. Keep vague phrases vague. You may change their grammar, but never make them more specific.
5. Write in the third person throughout, never "I", "my", "we" or "you". In a customer's \
message, complaint or email say "the writer" or "the customer" and what they reported or asked.
6. Keep who said what. A fact marked "(said by the agent)" is what the agent told the customer: \
write "The agent advised ..." or "The agent explained ...". Never turn it into something the \
customer said, did or plans.
7. Put the words around figures and product names in your own phrasing: never copy ten or \
more words in a row from a fact. Use as many short sentences as the facts need. Say each \
detail once.
8. Return only the rewritten record.

EXAMPLE 1
FACTS:
the customer - name: [PERSON_1].
the customer - ID number: [ID_1].
the customer - employer: a company.
the customer - income (said by the customer): earns 46,000 a month.
the customer - product (said by the customer): car loan.
the customer - amount (said by the customer): car loan instalment: 8,250.
the customer - issue (said by the customer): instalment taken twice in May.
the customer - account event (said by the customer): second debit bounced rent cheque.
the customer - plan (said by the customer): reverse the debit.
the customer - advice (said by the agent): wait five working days for the refund.
OUTPUT:
The customer, [PERSON_1], account [ID_1], works for a company and earns 46,000 a month. They \
reported that their car loan instalment of 8,250 was taken twice in May and that the second \
debit bounced their rent cheque, and asked for the debit to be reversed. The agent advised \
waiting five working days for the refund.

EXAMPLE 2
FACTS:
the record - amount: segment results, Nurseries (thousands): 2022: 6,614; 2021: 6,020.
the record - amount: segment results, Landscaping (thousands): 2022: 1,187; 2021: 1,342.
the record - other: Landscaping fell as two council contracts ended.
OUTPUT:
The segment results (thousands) for Nurseries were 6,614 in 2022 and 6,020 in 2021. The segment \
results (thousands) for Landscaping were 1,187 in 2022 and 1,342 in 2021. Landscaping fell as \
two council contracts ended.
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
        # whole words and whole numbers only: "12" must not hit the 12 of "4,312" or "12.5"
        span = r"\s+".join(re.escape(w) for w in d.span_text.split())
        span = rf"(?<!\w)(?<!\d[.,]){span}(?!\w)(?![.,]\d)"
        if d.action is Action.GENERALISE and re.search(span, value, re.IGNORECASE):
            # a vague phrase cannot replace a few words inside a longer one ("tore my anterior
            # cruciate ligament" -> "tore my a joint injury"); when the detail is most of the
            # value, the vague phrase is the whole value
            rest = re.sub(span, " ", value, flags=re.IGNORECASE)
            # ... unless the rest holds a figure: "4,318 in March 2016" keeps its 4,318
            if not re.search(r"\d", TOKEN.sub(" ", rest)) and 2 * len(
                _WORD.findall(d.span_text)
            ) >= len(_WORD.findall(TOKEN.sub(" ", value))):
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
        if d.category in _NAMED and _leftover(value, d.span_text, removed):
            assert d.replacement is not None
            if re.search(r"\d", TOKEN.sub(" ", value)):  # only the words go, the figure stays
                value = _without_leftover(value, d.span_text, d.replacement, removed)
            else:
                value = d.replacement
    value = DOUBLE_ARTICLE.sub(r"\1", value)  # "in a Pune branch" -> "in a branch"
    if not CASE_FILE.get():
        return value
    # a clinical record's date made vague to its year: "Review on 23/10/2026" -> "Review in 2026"
    return _ON_YEAR.sub(lambda m: ("In " if m.group(1) == "O" else "in ") + m.group(2), value)


DOUBLE_ARTICLE = re.compile(r"\b(?:an?|the)\s+(an?\s+\w)")
_ON_YEAR = re.compile(r"\b([Oo])n (\d{4}|the \d{4}s)\b")


_NAMED = {PIICategory.NAME, PIICategory.LOCATION, PIICategory.ADDRESS}
_GENERIC_PLACE_WORDS = {
    "hospital", "hospitals", "clinic", "clinics", "medical", "centre", "center", "health",
    "road", "street", "lane", "avenue", "nagar", "branch", "school", "college", "university",
    "apartments", "apartment", "flat", "house", "tower", "towers", "court", "block", "sector",
    "phase", "main", "cross", "north", "south", "east", "west", "city", "town", "district",
    "state", "the",
    # the ordinary words of an organisation's name: "Acme Home Loans Ltd" leaves "loans" alone
    "account", "annual", "associates", "bank", "business", "capital", "card", "company", "corp",
    "corporation", "credit", "energy", "enterprises", "finance", "financial", "global", "group",
    "holdings", "home", "housing", "income", "incorporated", "insurance", "interest", "limited",
    "loan", "loans", "mortgage", "national", "public", "savings", "securities", "services",
    "society", "trading", "trust", "union",
}  # fmt: skip


# The document being rewritten (set by ``agents.pipeline.sanitise``), so the leftover check
# below can tell a word of a removed name from the same word used elsewhere in the document.
SOURCE_TEXT: ContextVar[str | None] = ContextVar("source_text", default=None)


def _used_elsewhere(word: str, removed) -> bool:
    """True when the document also uses ``word`` outside every removed detail: "Imaging" in the
    title "Diagnostic Imaging Report" when the clinic "North Valley Imaging Center" was removed.
    Such a word is ordinary vocabulary of this document, not a trace of the removed name; a
    word found only inside removed details ("Valley", "Kengeri") still counts."""
    text = SOURCE_TEXT.get()
    if not text:
        return False
    cuts = [(d.start, d.end) for d in removed if d.start is not None and d.end is not None]
    for m in re.finditer(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", text):
        if not any(a <= m.start() < b for a, b in cuts):
            return True
    return False


def _distinctive(span_text: str, removed=()) -> set[str]:
    return {
        w
        for w in re.findall(r"[A-Z][A-Za-z'-]{3,}", span_text)
        if w.lower() not in _GENERIC_PLACE_WORDS and not _used_elsewhere(w, removed)
    }


def _without_leftover(value: str, span_text: str, replacement: str, removed=()) -> str:
    for word in sorted(_distinctive(span_text, removed), key=len, reverse=True):
        value = re.sub(rf"(?<![A-Za-z\[]){re.escape(word)}(?![A-Za-z])", replacement, value)
    return value


def _leftover(value: str, span_text: str, removed=()) -> bool:
    words = _distinctive(span_text, removed)
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
    # in a conversation with a bank the writer is "the customer"
    spoke = any(f.said_by == "customer" for f, _ in safe)
    writer = "the customer" if spoke else "the writer"
    for f, _ in safe:
        subject = f.subject
        if subject in labels:
            continue
        if subject == "patient":
            labels[subject] = "the patient"
        elif subject.startswith("patient."):
            labels[subject] = "the patient's " + subject.removeprefix("patient.").replace("_", " ")
        elif subject == "writer":
            labels[subject] = writer
        elif subject.startswith("writer."):
            labels[subject] = f"{writer}'s " + subject.removeprefix("writer.").replace("_", " ")
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


# facts about the record rather than about whoever wrote it
RECORD_RELATIONS = {"product", "issue", "amount", "account_event", "plan", "other"}
_FIGURE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def draft(facts: list[FactTuple], decisions: list[RoutingDecision]) -> str:
    """One line per fact the model may see, with removed details already replaced. A line is
    never repeated ("Laura Fenn" and "Laura" are one token, so one name line), and a person
    named by their token gets no separate line saying their name is that token."""
    safe = safe_facts(facts, decisions)
    labels = subject_labels(safe)
    lines: list[str] = []
    for f, value in safe:
        label = labels[f.subject]
        if f.subject == "writer" and f.relation in RECORD_RELATIONS and not f.said_by:
            label = "the record"
        if f.relation == "identifier" and value == label:
            continue
        said = f" (said by the {f.said_by})" if f.said_by else ""
        value = fidelity.third_person(value)
        relation = relation_label(f.relation, value)
        if f.said_by not in (None, "customer", "client") and f.relation == "plan":
            relation = "advice"  # a reply's plan is advice, not the customer's plan
        line = f"{label} - {relation}{said}: {value}."
        if line not in lines:
            lines.append(line)
    return "\n".join(lines)


def missing_figures(facts: list[FactTuple], decisions: list[RoutingDecision], text: str) -> list[str]:
    """Safe content facts with a figure the rewrite left out, as short lines to add. A long
    table is where a model drops one, and the figures are what questions ask for."""

    def figures(value: str) -> set[str]:
        return {m.replace(",", "") for m in _FIGURE.findall(TOKEN.sub(" ", value))}

    said = figures(text)
    lines: list[str] = []
    for f, value in safe_facts(facts, decisions):
        if f.relation not in RECORD_RELATIONS | {"income"} or figures(value) <= said:
            continue
        label = f.relation.replace("_", " ").capitalize()
        line = f"{label}: {fidelity.third_person(value)}."
        if line not in lines:
            lines.append(line)
    return lines


def build_user_prompt(
    facts: list[FactTuple],
    decisions: list[RoutingDecision],
    feedback: list[SensitiveSpan] | None = None,
    notes: list[str] | None = None,
) -> str:
    prompt = f"FACTS:\n{draft(facts, decisions) or '(no facts)'}"
    if feedback:
        kinds = ", ".join(sorted({s.category.value for s in feedback}))
        prompt += (
            f"\n\nYour previous version contained details that must not appear, or that are "
            f"not in the facts ({kinds}). Write it again using only the tokens and phrases in "
            "the facts."
        )
    if notes:
        prompt += "\n\nYour previous version was wrong:\n" + "\n".join(f"- {n}" for n in notes)
        prompt += "\nWrite it again."
    return prompt + "\nOUTPUT:"


async def reconstruct(
    facts: list[FactTuple],
    decisions: list[RoutingDecision],
    placeholders: dict[str, str],
    llm: LLMClient,
    feedback: list[SensitiveSpan] | None = None,
    notes: list[str] | None = None,
) -> str:
    system = CASE_FILE_PROMPT if is_case_file(facts) else SYSTEM_PROMPT
    reply = await llm.text_call(system, build_user_prompt(facts, decisions, feedback, notes))
    return PREAMBLE.sub("", reply).strip()
