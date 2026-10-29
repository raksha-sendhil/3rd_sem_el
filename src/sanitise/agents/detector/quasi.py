"""Stage 4 - LLM pass.  OWNER: Person B (Model and Prompt).

One constrained-JSON call per document. The model receives the raw text plus the
explicit identifiers the rule pass already found, and returns two kinds of item:

* **single details** - MEDICAL_CONDITION, SYMPTOM, OCCUPATION and named LOCATION
  (institutions, small places), which rules cannot enumerate, plus any NAME or AGE
  the rule pass missed (uncommon names in weak contexts, ages written oddly). These
  must be quoted verbatim; ``ground()`` finds every occurrence in the document and
  gives each one exact offsets, so they are scored and later rewritten like rule
  spans. A detail that cannot be found in the text is a hallucination and is dropped.
* **QUASI_COMBO** - a combination of details that together narrow the candidate
  pool (spec 3.1 stage 4). These have no single position, so start/end stay None.

Decoding is constrained to ``LLMQuasiList``: the same shape as the frozen
``QuasiList`` (and the same JSON-schema title, so MockLLM scripts keyed on
"QuasiList" keep working) but with the category enum narrowed to the seven values
above. The model therefore cannot emit DATE for "pain" or RELATION for a job,
which were the main failure modes seen with llama3:8b on Day 6. (NAME and AGE were
added after a narrower enum forced the model to label a name it had recognised as
OCCUPATION.)

Parsing, validation and the retry/repair loop (up to ``retries`` extra attempts
with escalating temperature and the validation error fed back) are done once for
every schema by ``sanitise.llm.base.BaseJSONClient``; this module only adds the
prompt, the few-shot examples and the post-processing that a schema cannot express.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sanitise.llm.base import LLMClient
from sanitise.schemas import PIICategory, SensitiveSpan

log = logging.getLogger(__name__)

C = PIICategory

# Categories the LLM pass may produce. Everything else is the rule pass's job.
LLMCategory = Literal[
    "NAME", "AGE", "MEDICAL_CONDITION", "SYMPTOM", "OCCUPATION", "LOCATION", "QUASI_COMBO"
]
GROUNDED = {C.NAME, C.AGE, C.MEDICAL_CONDITION, C.SYMPTOM, C.OCCUPATION, C.LOCATION}
MAX_COMBOS = 3  # the gold set never has more than 2 per document
MAX_ITEMS = 20


class LLMQuasiHit(BaseModel):
    text: str = Field(min_length=1, max_length=300)
    category: LLMCategory
    rationale: str = Field(min_length=1, max_length=500)


class LLMQuasiList(BaseModel):
    """The schema handed to Ollama (``format``). Titled "QuasiList" on purpose."""

    model_config = ConfigDict(title="QuasiList")
    quasi_identifiers: list[LLMQuasiHit]


# ------------------------------------------------------------------ prompt

SYSTEM_PROMPT = """\
You are a privacy analyst. You read one document (a patient's question to a doctor, \
or a workplace email) and list the details that could re-identify the person it is about.
A rule-based detector has ALREADY found most explicit identifiers (names, ages, dates, \
phone numbers, emails, ID numbers, addresses, relations, body measurements). They are \
listed under ALREADY FOUND. Never repeat anything listed there. Do not return dates, \
phone numbers, emails, ID numbers, addresses, relations or body measurements. The \
detector does sometimes miss NAMES and AGES, so return those when they are not \
ALREADY FOUND.

Return two kinds of item.

1. Single details, copied EXACTLY as written in the document (same words, no article, \
shortest phrase that carries the detail):
   - NAME: a person's name (first name, surname or both) without a title such as Dr. \
or Mr. Not the name of a company, hospital, drug or disease.
   - AGE: a person's age as written, including in words ("eleven", "61 year old"). \
Not a duration ("two weeks").
   - MEDICAL_CONDITION: a named diagnosis, disease or specific clinical finding about \
a specific person, including a suspected one ("Could this be gout?"). NOT plain \
symptoms (those are SYMPTOM), NOT procedures or surgeries, NOT drugs, NOT general \
statements about a disease.
   - SYMPTOM: a symptom or complaint that a specific person in the document has \
(fever, nausea, sore throat, itching, numbness). Include the body part it affects \
("pain in the left knee", "itching on both arms"); leave out severity words such as \
mild or severe, and durations. NOT a symptom that is negated or only asked about \
("no chest pain", "should I worry if I start coughing?").
   - OCCUPATION: the job or role of a person in the document ("crane operator", \
"PhD student", "night-shift baker"). Keep qualifiers that make it rarer.
   - LOCATION: a named place that narrows where the person is - a town, city, island, \
hospital, clinic, school, university or named workplace site - unless it is ALREADY FOUND.

2. QUASI_COMBO: a combination of details about the same person that together narrow \
the candidate pool to a small group, when each detail alone would not. Only when at \
least TWO of these apply: a rare or niche occupation; a rare condition or an unusual \
combination of symptoms; a small or specific location or named institution; a precise \
timeline (an exact date or duration tied to a place or event); an extreme value (age 90+, \
unusual height). The text is a short description that names the parts, joined with \
" + ", for example "rare occupation (glassblower) + small island (Murano) + sarcoidosis". \
At most two combinations; one per distinct combination. Common combinations (an adult \
with diabetes in a big city) are not quasi-identifiers.

Every item needs a one-sentence rationale saying why it identifies the person.
If nothing applies, return {"quasi_identifiers": []}. Do not invent details that are \
not in the document.

EXAMPLE 1
DOCUMENT:
Hello doctor, I am a 45 year old glassblower in Murano. Last year I was diagnosed with \
sarcoidosis and now I get a hoarse voice at night. My wife thinks it is the furnace smoke.
ALREADY FOUND:
- AGE: 45 year old
- DATE: Last year
- RELATION: My wife
OUTPUT:
{"quasi_identifiers": [
 {"text": "glassblower", "category": "OCCUPATION", "rationale": "The writer's job, a rare trade."},
 {"text": "Murano", "category": "LOCATION", "rationale": "A small island where the writer works."},
 {"text": "sarcoidosis", "category": "MEDICAL_CONDITION", \
"rationale": "A named diagnosis for the writer."},
 {"text": "hoarse voice", "category": "SYMPTOM", "rationale": "A symptom the writer has."},
 {"text": "rare occupation (glassblower) + small island (Murano) + sarcoidosis", \
"category": "QUASI_COMBO", "rationale": "Few glassblowers on one small island have \
sarcoidosis, so together these point to one person."}
]}
(The hoarse voice is a symptom, so it is SYMPTOM, not MEDICAL_CONDITION.)

EXAMPLE 2
DOCUMENT:
Hi team, the Q3 pipeline numbers are attached. I will be out on Friday for my \
daughter's graduation. Thanks, Raj
ALREADY FOUND:
- DATE: Friday
- RELATION: my daughter
- NAME: Raj
OUTPUT:
{"quasi_identifiers": []}

EXAMPLE 3
DOCUMENT:
My 7 year old son goes to Sunrise Montessori in Kodaikanal. Since 2 March he has been very \
thirsty and the paediatrician at Nilgiri Hospital says it is type 1 diabetes.
ALREADY FOUND:
- AGE: 7 year old
- RELATION: My son
- LOCATION: Kodaikanal
- DATE: 2 March
OUTPUT:
{"quasi_identifiers": [
 {"text": "Sunrise Montessori", "category": "LOCATION", \
"rationale": "The named school the child attends."},
 {"text": "Nilgiri Hospital", "category": "LOCATION", \
"rationale": "The named hospital treating the child."},
 {"text": "thirsty", "category": "SYMPTOM", "rationale": "The child's symptom."},
 {"text": "type 1 diabetes", "category": "MEDICAL_CONDITION", \
"rationale": "The child's diagnosis."},
 {"text": "child age + named school + small town (Kodaikanal) + exact onset date", \
"category": "QUASI_COMBO", "rationale": "One 7-year-old at a named school in a small \
town with a known onset date is easy to single out."}
]}

EXAMPLE 4
DOCUMENT:
My daughter Ishani is nine and has had eczema since 2021. Her doctor, Dr. Mehra, \
works in the paediatrics ward of Lakeside Clinic.
ALREADY FOUND:
- RELATION: My daughter
- DATE: 2021
- NAME: Mehra
OUTPUT:
{"quasi_identifiers": [
 {"text": "Ishani", "category": "NAME", "rationale": "The daughter's first name."},
 {"text": "nine", "category": "AGE", "rationale": "The daughter's age, written in words."},
 {"text": "eczema", "category": "MEDICAL_CONDITION", "rationale": "The daughter's diagnosis."},
 {"text": "Lakeside Clinic", "category": "LOCATION", \
"rationale": "The named clinic treating her."}
]}
(Eczema in a child is common, so there is no QUASI_COMBO. The paediatrics ward is part \
of the clinic, not anyone's job. Mehra is ALREADY FOUND, so it is not repeated.)
"""


TASK_REMINDER = (
    "List any NAME or AGE that is NOT in ALREADY FOUND, the MEDICAL_CONDITION (diagnoses and "
    "clinical findings), SYMPTOM (symptoms the person has), OCCUPATION and LOCATION details "
    "in this document that are NOT in ALREADY FOUND, then any QUASI_COMBO combinations."
)


def build_user_prompt(text: str, explicit: list[SensitiveSpan]) -> str:
    found = "\n".join(f"- {s.category.value}: {s.text}" for s in explicit) or "(none)"
    return f"DOCUMENT:\n{text}\n\nALREADY FOUND:\n{found}\n\n{TASK_REMINDER}\nOUTPUT:"


# ------------------------------------------------------------------ full detection
# A strong model (``one_call_per_document``, e.g. Gemini) does the whole detection itself:
# every category, every person, with the rule pass's spans as hints only. The narrow prompt
# above stays for the small local models it was written for.

FullCategory = Literal[
    "NAME", "AGE", "DATE", "PHONE", "EMAIL", "ID_NUMBER", "ADDRESS", "LOCATION",
    "OCCUPATION", "MEDICAL_CONDITION", "SYMPTOM", "RELATION", "PHYSICAL", "OTHER",
    "QUASI_COMBO",
]  # fmt: skip
MAX_FULL_ITEMS = 200
# A RELATION names a family or personal tie. The model has tagged the "team" of "Hi team" as one,
# which then became "the writer's team" in Agent 2; a span without one of these words is dropped.
RELATION_WORDS = {
    "wife", "husband", "spouse", "partner", "son", "daughter", "child", "children", "kid", "kids",
    "baby", "boy", "girl", "mother", "father", "mom", "mum", "mommy", "mummy", "mama", "dad",
    "daddy", "papa", "parent", "parents", "brother", "sister", "sibling", "siblings", "bro", "sis",
    "grandmother", "grandfather", "grandma", "grandpa", "granny", "nana", "grandson",
    "granddaughter", "grandchild", "aunt", "auntie", "uncle", "niece", "nephew", "cousin",
    "fiance", "fiancee", "girlfriend", "boyfriend", "friend", "relative", "family", "guardian",
    "stepmother", "stepfather", "stepson", "stepdaughter", "in-law", "in-laws",
}  # fmt: skip


class FullHit(BaseModel):
    text: str = Field(min_length=1, max_length=300)
    category: FullCategory
    rationale: str = Field(min_length=1, max_length=200)


class FullDetection(BaseModel):
    model_config = ConfigDict(title="FullDetection")
    details: list[FullHit]


FULL_SYSTEM_PROMPT = """\
You are a privacy analyst. You read one document (a patient's message to a doctor, an \
email, a note) and list EVERY detail that could help identify ANY person in it: the \
writer, the people it is sent to or copied to, and everyone it mentions. Email headers \
(From, To, Cc, Sent), signatures and quoted earlier messages count like the body.

Categories (copy each detail EXACTLY as written, shortest phrase that carries it):
- NAME: a person's name, full or partial, nickname, or a username that is a person's name. \
Not titles (Dr., Mr.), companies, products, hospitals or diseases.
- AGE: a person's age as written ("45", "61 year old", "nine").
- DATE: a specific date or day ("4/19", "June 21-22", "Friday", "Sep 24th", "last May"). \
Not vague times ("recently", "two weeks").
- PHONE: phone and fax numbers, including short internal extensions ("3-5620", "x31647").
- EMAIL: email addresses.
- ID_NUMBER: account, member, policy, case, employee or other reference numbers and \
account usernames.
- ADDRESS: street addresses, room or office numbers, P.O. boxes, postcodes.
- LOCATION: a named place: city, town, state, country, hospital, clinic, school, \
university, hotel, workplace site.
- OCCUPATION: a person's job, title or role ("crane operator", "Senior Counsel", "student").
- MEDICAL_CONDITION: a diagnosis, disease or clinical finding someone has or may have, \
including suspected ones and pregnancy.
- SYMPTOM: a symptom or complaint someone has, with the body part when given \
("back pain", "coughing", "pain in the left knee"). Not a negated or hypothetical one.
- RELATION: a family or personal relationship to someone ("my son", "Dad", "his wife"). \
Not a group addressed in a greeting or at work ("Hi team", "all", "everyone", "colleagues").
- PHYSICAL: height, weight or another body measurement.
- OTHER: any other secret or identifying value (a password, a licence plate).
- QUASI_COMBO: at most two combinations of details about one person that together \
single them out when each alone would not, written as parts joined with " + ".

HINTS lists what a rule-based detector found. It may be incomplete and sometimes wrong: \
include the hints that are right (with the right category) and add everything it missed. \
Return each distinct detail once; every place it occurs is marked automatically. \
Rationale: a few words. Do not invent details that are not in the document.

EXAMPLE 1
DOCUMENT:
From: Lena Ortiz
Sent: Tuesday, March 4, 2003 9:12 AM
To: Bram Tuck; Kofi
Subject: site visit

Bram - the inspector from Dallas moved our visit to 3/11. Kofi, can you book the \
Riverside Inn? Call me at 555-0142 or ext. 4471.
Lena
Senior Engineer, 77 Harbor Rd, Suite 4B
OUTPUT:
{"details": [
 {"text": "Lena Ortiz", "category": "NAME", "rationale": "sender"},
 {"text": "Lena", "category": "NAME", "rationale": "sender's sign-off"},
 {"text": "Tuesday, March 4, 2003", "category": "DATE", "rationale": "sent date"},
 {"text": "Bram Tuck", "category": "NAME", "rationale": "recipient"},
 {"text": "Bram", "category": "NAME", "rationale": "recipient"},
 {"text": "Kofi", "category": "NAME", "rationale": "recipient"},
 {"text": "inspector", "category": "OCCUPATION", "rationale": "a person's role"},
 {"text": "Dallas", "category": "LOCATION", "rationale": "inspector's city"},
 {"text": "3/11", "category": "DATE", "rationale": "visit date"},
 {"text": "Riverside Inn", "category": "LOCATION", "rationale": "named hotel"},
 {"text": "555-0142", "category": "PHONE", "rationale": "sender's phone"},
 {"text": "ext. 4471", "category": "PHONE", "rationale": "sender's extension"},
 {"text": "Senior Engineer", "category": "OCCUPATION", "rationale": "sender's title"},
 {"text": "77 Harbor Rd, Suite 4B", "category": "ADDRESS", "rationale": "work address"}
]}

EXAMPLE 2
DOCUMENT:
Hi doctor, my mother is 68 and since 2nd May she has had swelling in both ankles and \
gets breathless climbing stairs. She had a heart attack in 2019 at St. Anne's in Pune. \
She is 5 ft 1 and was a schoolteacher. Is it heart failure?
OUTPUT:
{"details": [
 {"text": "my mother", "category": "RELATION", "rationale": "the patient"},
 {"text": "68", "category": "AGE", "rationale": "mother's age"},
 {"text": "2nd May", "category": "DATE", "rationale": "onset date"},
 {"text": "swelling in both ankles", "category": "SYMPTOM", "rationale": "mother's symptom"},
 {"text": "breathless climbing stairs", "category": "SYMPTOM", "rationale": "mother's symptom"},
 {"text": "heart attack", "category": "MEDICAL_CONDITION", "rationale": "past diagnosis"},
 {"text": "2019", "category": "DATE", "rationale": "year of the heart attack"},
 {"text": "St. Anne's", "category": "LOCATION", "rationale": "named hospital"},
 {"text": "Pune", "category": "LOCATION", "rationale": "city"},
 {"text": "5 ft 1", "category": "PHYSICAL", "rationale": "height"},
 {"text": "schoolteacher", "category": "OCCUPATION", "rationale": "mother's former job"},
 {"text": "heart failure", "category": "MEDICAL_CONDITION", "rationale": "suspected diagnosis"}
]}
"""


def build_full_prompt(text: str, hints: list[SensitiveSpan]) -> str:
    found = "\n".join(f"- {s.category.value}: {s.text}" for s in hints) or "(none)"
    return (
        f"DOCUMENT:\n{text}\n\nHINTS:\n{found}\n\n"
        "List every identifying detail about every person in this document.\nOUTPUT:"
    )


# ------------------------------------------------------------------ post-processing

_LEADING = re.compile(r"^(?:a|an|the|my|his|her|their|our)\s+", re.IGNORECASE)
_EDGE_PUNCT = " \t\n\"'`.,;:!?()[]{}"


def _clean(phrase: str) -> str:
    phrase = phrase.strip(_EDGE_PUNCT)
    while (stripped := _LEADING.sub("", phrase)) != phrase:
        phrase = stripped
    return phrase.strip(_EDGE_PUNCT)


def find_all(text: str, phrase: str) -> list[tuple[int, int]]:
    """Every whole-word, case-insensitive occurrence of ``phrase`` in ``text``.

    Internal whitespace may differ (the model often collapses line breaks), so
    the phrase's words are joined with ``\\s+``.
    """
    words = phrase.split()
    if not words:
        return []
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(w) for w in words) + r"(?!\w)"
    return [(m.start(), m.end()) for m in re.finditer(pattern, text, re.IGNORECASE)]


def ground(text: str, phrase: str) -> list[tuple[int, int]]:
    """Offsets for a single detail quoted by the model; [] when it is not in the text.

    The phrase is tried without a leading article/possessive first (annotation guide
    3.7: "lighthouse keeper", not "a lighthouse keeper"), then exactly as quoted.
    """
    cleaned = _clean(phrase)
    return (find_all(text, cleaned) if cleaned else []) or find_all(text, phrase)


def to_spans(
    text: str, hits: list[LLMQuasiHit] | list[FullHit], max_items: int = MAX_ITEMS
) -> list[SensitiveSpan]:
    """Ground single details to offsets, keep combinations offset-less, deduplicate."""
    spans: list[SensitiveSpan] = []
    seen_offsets: set[tuple[int, int]] = set()
    seen_combos: set[str] = set()
    for hit in hits[:max_items]:
        category = C(hit.category)
        rationale = hit.rationale.strip() or "flagged by the LLM pass"
        if category is C.QUASI_COMBO:
            key = " ".join(hit.text.lower().split())
            if key in seen_combos or len(seen_combos) >= MAX_COMBOS:
                continue
            seen_combos.add(key)
            spans.append(
                SensitiveSpan(
                    text=hit.text.strip(), category=category, method="llm", rationale=rationale
                )
            )
            continue
        offsets = ground(text, hit.text)
        if not offsets:
            log.info("LLM %s %r is not in the document; dropped", category.value, hit.text)
            continue
        for start, end in offsets:
            if (start, end) in seen_offsets:
                continue
            seen_offsets.add((start, end))
            spans.append(
                SensitiveSpan(
                    text=text[start:end],
                    category=category,
                    start=start,
                    end=end,
                    method="llm",
                    rationale=rationale,
                )
            )
    return spans


# ------------------------------------------------------------------ the seam


async def infer_quasi(
    text: str, explicit: list[SensitiveSpan], llm: LLMClient
) -> list[SensitiveSpan]:
    """One constrained-JSON call. Returns spans with method='llm'.

    QUASI_COMBO spans have start/end None (spec 8.2). Single details
    (NAME, AGE, MEDICAL_CONDITION, SYMPTOM, OCCUPATION, LOCATION) carry the offsets of every place
    the quoted phrase occurs in ``text``. A strong model gets the full-detection prompt instead:
    every category, with the rule spans as hints.
    """
    if getattr(llm, "one_call_per_document", False):
        full = await llm.json_call(
            FULL_SYSTEM_PROMPT, build_full_prompt(text, explicit), FullDetection
        )
        details = [
            d
            for d in full.details
            if d.category != "RELATION"
            or RELATION_WORDS & set(re.findall(r"[a-z]+(?:-[a-z]+)*", d.text.lower()))
        ]
        return to_spans(text, details, MAX_FULL_ITEMS)
    result = await llm.json_call(SYSTEM_PROMPT, build_user_prompt(text, explicit), LLMQuasiList)
    return to_spans(text, result.quasi_identifiers)
