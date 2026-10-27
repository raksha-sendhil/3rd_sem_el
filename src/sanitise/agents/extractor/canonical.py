"""Agent 2 - subjects and fact normalisation.  OWNER: Person A (Deterministic Logic).

``subjects(text, spans)`` finds every person a document talks about and gives each a stable id,
so the fact-extraction prompt can attach every fact to the right person:

* ``writer`` - the author. Mentions: first-person words as written ("I", "my"), and the
  writer's own name when the text introduces it ("my name is ...", "I am ...", "this is ...",
  an email ``From:`` line, or a name alone on a sign-off line).
* ``writer.<relation>`` - a relative of the writer, from a RELATION span with "my" / "our" or a
  bare form of address ("Mom"): ``My son`` -> ``writer.son``, ``my mother-in-law`` ->
  ``writer.mother_in_law``. A name right next to it ("My father, Cyril D'Souza,") joins it.
* ``person_<n>`` - anyone else: a name that is not the writer's, or a relative of someone else
  ("His wife" -> the wife of whoever "His" refers to). A first name joins the full name it is
  part of ("Laura" and "Laura Fenn" are one person).

Third-person pronouns (he / him / his / she / her) are added to the mentions of the nearest
person before them whose gender does not contradict the pronoun (kinship words carry a gender,
names do not). Mentions are exact strings from the text, without duplicates, in order of
first appearance.

``normalise_facts(facts, subjects)`` makes the model's facts consistent:

* the subject may come back as an id, a label or a mention ("the writer's son", "My son",
  "Priya", "patient"); it is mapped to the subject's id, and facts whose subject matches no
  one are dropped;
* relation synonyms are mapped onto ``schemas.RELATIONS`` (``normalise_relation``), anything
  unrecognised becomes ``other``;
* values are trimmed (whitespace collapsed, surrounding quotes and trailing punctuation
  removed) and empty ones dropped;
* duplicates (same subject, relation and value, ignoring case, spacing, punctuation and a
  leading article) are merged into the first one, keeping the highest importance.

Everything is deterministic and needs no model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sanitise.schemas import RELATIONS, FactTuple, Importance, PIICategory, SensitiveSpan, Subject

C = PIICategory

# ---------------------------------------------------------------- relations of kinship

_POSSESSIVE = re.compile(r"^(my|our|his|her|their|your)\s+", re.IGNORECASE)
_MODIFIERS = {
    "little", "baby", "infant", "newborn", "elder", "older", "younger", "eldest", "youngest",
    "only", "teenage", "adult", "late", "ex", "dear", "beloved", "old", "year", "years",
    "month", "months", "week", "weeks",
}  # fmt: skip
_KIN_SYNONYMS = {
    "mom": "mother", "mum": "mother", "mommy": "mother", "mummy": "mother", "mama": "mother",
    "ma": "mother", "dad": "father", "daddy": "father", "papa": "father", "pa": "father",
    "grandma": "grandmother", "granny": "grandmother", "nana": "grandmother",
    "grandpa": "grandfather", "hubby": "husband", "kid": "child", "kids": "children",
    "auntie": "aunt", "babies": "baby", "boy": "son", "girl": "daughter",
}  # fmt: skip
_MALE = {
    "son", "husband", "father", "brother", "grandfather", "grandson", "uncle", "nephew",
    "boyfriend", "stepson", "stepfather", "son_in_law", "father_in_law", "brother_in_law",
}  # fmt: skip
_FEMALE = {
    "daughter", "wife", "mother", "sister", "grandmother", "granddaughter", "aunt", "niece",
    "girlfriend", "stepdaughter", "stepmother", "daughter_in_law", "mother_in_law",
    "sister_in_law",
}  # fmt: skip


def parse_relation(span_text: str) -> tuple[str, str | None]:
    """``"My 5 year old son"`` -> ``("son", "my")``; ``"Mom"`` -> ``("mother", None)``."""
    text = " ".join(span_text.strip().split())
    m = _POSSESSIVE.match(text)
    possessive = m.group(1).lower() if m else None
    rest = text[m.end() :] if m else text
    rest = re.sub(r"[\s-]+in[\s-]+laws?\b", "_in_law", rest.lower())
    words = re.findall(r"[a-z_]+", rest)
    kept = [w for w in words if w not in _MODIFIERS] or words[-1:]
    if not kept:
        return "relative", possessive
    return _KIN_SYNONYMS.get(kept[-1], kept[-1]), possessive


def _gender(kin: str) -> str | None:
    return "m" if kin in _MALE else "f" if kin in _FEMALE else None


# ---------------------------------------------------------------- subjects

_WRITER_NAME_BEFORE = re.compile(
    r"(?:my\s+name\s+is|i\s+am|i'm|this\s+is|from:|signed,?)\s*$", re.IGNORECASE
)
_SIGN_OFF = re.compile(
    r"^\s*(?:thanks|thank\s+you|regards|best|best\s+regards|kind\s+regards|cheers|sincerely"
    r"|love|warm\s+regards|yours(?:\s+\w+)?)\s*[,.!]?\s*$",
    re.IGNORECASE,
)
_FIRST_PERSON = re.compile(r"\b(?:I|me|my|My|mine|myself)\b")
_PRONOUN = re.compile(r"\b(?:he|him|his|she|her|hers)\b", re.IGNORECASE)
_NAME_TOKEN = re.compile(r"[A-Za-z][A-Za-z'\u2019.-]*")


@dataclass
class _Subject:
    id: str
    label: str
    gender: str | None = None
    mentions: list[str] = field(default_factory=list)
    names: list[set[str]] = field(default_factory=list)  # lower-cased tokens of each name

    def add(self, mention: str) -> None:
        if mention and mention not in self.mentions:
            self.mentions.append(mention)


def _name_tokens(name: str) -> set[str]:
    return {t.lower().strip(".'\u2019") for t in _NAME_TOKEN.findall(name) if len(t.strip(".")) > 1}


def _is_writer_name(text: str, span: SensitiveSpan) -> bool:
    assert span.start is not None and span.end is not None
    before = text[max(0, span.start - 40) : span.start]
    if _WRITER_NAME_BEFORE.search(before):
        return True
    # alone on its line, after a sign-off line (or as the last line of the document)
    line_start = text.rfind("\n", 0, span.start) + 1
    line_end = text.find("\n", span.end)
    line_end = len(text) if line_end < 0 else line_end
    if text[line_start:line_end].strip(" ,.") != span.text:
        return False
    previous = [ln for ln in text[:line_start].splitlines() if ln.strip()]
    rest = text[line_end:].strip()
    return bool(previous and _SIGN_OFF.match(previous[-1])) or (
        not rest and len(previous) > 1 and not previous[-1].rstrip().endswith((":", ","))
    )


def subjects(text: str, spans: list[SensitiveSpan]) -> list[Subject]:
    writer = _Subject("writer", "the writer")
    found: list[_Subject] = [writer]
    by_id = {"writer": writer}
    # (start, subject) for every place a non-writer subject is mentioned, for pronouns
    anchors: list[tuple[int, _Subject]] = []
    counter = 0

    def new_person(label: str, gender: str | None = None) -> _Subject:
        nonlocal counter
        counter += 1
        subject = _Subject(f"person_{counter}", label, gender)
        found.append(subject)
        by_id[subject.id] = subject
        return subject

    def antecedent(pos: int, gender: str | None) -> _Subject | None:
        for start, subject in reversed(anchors):
            if start < pos and (gender is None or subject.gender in (None, gender)):
                return subject
        return None

    ordered = sorted(
        (s for s in spans if s.start is not None and s.category in (C.RELATION, C.NAME)),
        key=lambda s: s.start or 0,
    )
    previous: tuple[SensitiveSpan, _Subject] | None = None
    for span in ordered:
        assert span.start is not None and span.end is not None
        gap = text[previous[0].end : span.start] if previous else None
        adjacent = gap is not None and re.fullmatch(r"\s*,?\s*", gap) is not None
        if span.category is C.RELATION:
            kin, possessive = parse_relation(span.text)
            gender = _gender(kin)
            if possessive in (None, "my", "our"):
                sid = f"writer.{kin}"
                subject = by_id.get(sid)
                if subject is None:
                    subject = _Subject(sid, f"the writer's {kin.replace('_', ' ')}", gender)
                    # "Linda, my wife": the name just before is this relative
                    if previous and adjacent and previous[0].category is C.NAME:
                        named = previous[1]
                        if named.id.startswith("person_") and len(named.mentions) == 1:
                            found.remove(named)
                            by_id.pop(named.id)
                            anchors[:] = [(p, subject if s is named else s) for p, s in anchors]
                            subject.mentions, subject.names = named.mentions, named.names
                    found.append(subject)
                    by_id[sid] = subject
            else:
                owner = antecedent(span.start, {"his": "m", "her": "f"}.get(possessive or ""))
                owner_label = owner.label if owner else "someone"
                subject = new_person(f"{owner_label}'s {kin.replace('_', ' ')}", gender)
            subject.add(span.text)
            anchors.append((span.start, subject))
        else:  # NAME
            tokens = _name_tokens(span.text)
            if previous and adjacent and previous[0].category is C.RELATION:
                subject = previous[1]
            elif _is_writer_name(text, span):
                subject = writer
            else:
                subject = next(
                    (
                        s
                        for s in found
                        if any(tokens and (tokens <= n or n <= tokens) for n in s.names)
                    ),
                    None,
                ) or new_person(span.text)
            subject.add(span.text)
            subject.names.append(tokens)
            if subject is not writer:
                anchors.append((span.start, subject))
        previous = (span, subject)

    # "His" in "His wife" is part of a relation, not a mention of someone
    inside = [
        (s.start, s.end)
        for s in spans
        if s.start is not None and s.end is not None and s.category in (C.RELATION, C.NAME)
    ]
    for m in _PRONOUN.finditer(text):
        if any(a <= m.start() < b for a, b in inside):
            continue
        word = m.group().lower()
        gender = "m" if word in ("he", "him", "his") else "f"
        subject = antecedent(m.start(), gender)
        if subject is not None:
            subject.add(m.group())
    for m in _FIRST_PERSON.finditer(text):
        if not any(a <= m.start() < b for a, b in inside):
            writer.add(m.group())

    return [Subject(id=s.id, label=s.label, mentions=s.mentions) for s in found]


# ---------------------------------------------------------------- relations of facts

_RELATION_SYNONYMS = {
    "aged": "age", "years_old": "age", "age_years": "age",
    "gender": "sex",
    "lives": "lives_in", "lives_at": "lives_in", "resides_in": "lives_in", "resides": "lives_in",
    "location": "lives_in", "residence": "lives_in", "hometown": "lives_in", "city": "lives_in",
    "home": "lives_in", "address": "lives_in", "from": "lives_in", "based_in": "lives_in",
    "occupation": "works_as", "job": "works_as", "profession": "works_as", "works": "works_as",
    "employed_as": "works_as", "role": "works_as", "job_title": "works_as", "works_at": "works_as",
    "studies": "studies_at", "school": "studies_at", "college": "studies_at",
    "university": "studies_at", "student_at": "studies_at", "attends": "studies_at",
    "hospital": "treated_at", "treated_in": "treated_at", "admitted_to": "treated_at",
    "admitted_at": "treated_at", "seen_at": "treated_at", "clinic": "treated_at",
    "condition": "has_condition", "diagnosis": "has_condition", "diagnosed_with": "has_condition",
    "disease": "has_condition", "illness": "has_condition", "has_disease": "has_condition",
    "suffers_from": "has_condition", "medical_condition": "has_condition",
    "symptom": "has_symptom", "symptoms": "has_symptom", "complains_of": "has_symptom",
    "experiences": "has_symptom", "experiencing": "has_symptom", "has_symptoms": "has_symptom",
    "duration": "symptom_duration", "symptom_since": "symptom_duration",
    "since": "symptom_duration", "for_how_long": "symptom_duration",
    "medication": "took_medication", "takes_medication": "took_medication",
    "takes": "took_medication", "drug": "took_medication", "medicine": "took_medication",
    "prescribed": "took_medication", "takes_medicine": "took_medication",
    "procedure": "had_procedure", "surgery": "had_procedure", "operation": "had_procedure",
    "underwent": "had_procedure", "test": "had_procedure", "had_test": "had_procedure",
    "date": "event_date", "when": "event_date", "onset_date": "event_date",
    "diagnosed_on": "event_date", "admitted_on": "event_date",
    "relationship": "relation_to", "relative_of": "relation_to", "related_to": "relation_to",
    "relation": "relation_to", "kinship": "relation_to",
    "tall": "height", "weighs": "weight", "body_weight": "weight",
    "phone": "contact", "email": "contact", "mobile": "contact", "contact_number": "contact",
    "phone_number": "contact", "telephone": "contact",
    "id": "identifier", "name": "identifier", "patient_id": "identifier",
    "policy_number": "identifier", "employee_id": "identifier", "id_number": "identifier",
}  # fmt: skip

# (token prefixes, relation), checked in order on the tokens of an unknown relation name
_RELATION_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (("duration", "long", "since", "lasting"), "symptom_duration"),
    (("symptom", "complain", "experienc", "feel", "pain"), "has_symptom"),
    (("medic", "drug", "prescri", "tablet", "dose", "pill"), "took_medication"),
    (("surger", "procedure", "operat", "biopsy", "scan", "therap", "test"), "had_procedure"),
    (("diagnos", "condition", "disease", "illness", "disorder"), "has_condition"),
    (("date", "when", "onset", "day", "admitted", "discharged"), "event_date"),
    (("hospital", "clinic", "treat"), "treated_at"),
    (("stud", "school", "college", "universit", "student"), "studies_at"),
    (("work", "job", "occupation", "profession", "employ", "career"), "works_as"),
    (("live", "resid", "home", "town", "city", "location", "address", "place"), "lives_in"),
    (("phone", "email", "contact", "mobile"), "contact"),
    (("identifier", "id", "number", "name"), "identifier"),
    (("height", "tall"), "height"),
    (("weight", "weigh"), "weight"),
    (("gender", "sex"), "sex"),
    (("age", "old"), "age"),
    (("relat", "family", "kin", "parent", "spouse"), "relation_to"),
]


def normalise_relation(relation: str) -> str:
    """Map a relation name onto ``schemas.RELATIONS``; ``other`` when nothing fits."""
    key = re.sub(r"[^a-z0-9]+", "_", relation.lower()).strip("_")
    key = key.removeprefix("the_").removeprefix("is_")
    if key in RELATIONS:
        return key
    if key in _RELATION_SYNONYMS:
        return _RELATION_SYNONYMS[key]
    tokens = [t for t in key.split("_") if t]
    for prefixes, target in _RELATION_KEYWORDS:
        if any(t.startswith(p) for t in tokens for p in prefixes):
            return target
    return "other"


# ---------------------------------------------------------------- facts

_WRITER_ALIASES = {
    "writer", "the writer", "author", "the author", "i", "me", "my", "myself", "self",
    "patient", "the patient", "user", "the user", "sender", "the sender", "narrator",
}  # fmt: skip
_ARTICLE = re.compile(r"^(?:a|an|the|my|his|her|their|our)\s+", re.IGNORECASE)
_RANK = {Importance.LOW: 0, Importance.MEDIUM: 1, Importance.HIGH: 2}


def _key(text: str) -> str:
    return " ".join(re.sub(r"[^\w']+", " ", text.lower()).split())


def _subject_lookup(subjects: list[Subject]) -> dict[str, str]:
    lookup: dict[str, str] = {alias: "writer" for alias in _WRITER_ALIASES}
    kin_count: dict[str, list[str]] = {}
    for s in subjects:
        for alias in (s.id, s.label, s.label.removeprefix("the ")):
            lookup[_key(alias)] = s.id
        for mention in s.mentions:
            if mention.lower() not in {"he", "him", "his", "she", "her", "hers"}:
                lookup.setdefault(_key(mention), s.id)
        if s.id.startswith("writer."):
            kin = s.id.removeprefix("writer.")
            lookup.setdefault(f"writer's {kin.replace('_', ' ')}", s.id)
            kin_count.setdefault(kin.replace("_", " "), []).append(s.id)
    for kin, ids in kin_count.items():
        if len(ids) == 1:
            lookup.setdefault(kin, ids[0])
    return lookup


_WRITER_KIN = re.compile(r"^(?:the\s+)?writer(?:'s|s)?[\s_.]+(.+)$")


def _resolve_subject(subject: str, lookup: dict[str, str], ids: set[str]) -> str | None:
    """An id, label or mention -> the subject's id; ``None`` if it matches no one."""
    if subject.strip() in ids:
        return subject.strip()
    key = _key(subject)
    for candidate in (key, _ARTICLE.sub("", key)):
        if candidate in lookup:
            return lookup[candidate]
    m = _WRITER_KIN.match(subject.strip().lower())  # "writer son", "writer_mother", "writers son"
    if m:
        kin = re.sub(r"[\s-]+", "_", m.group(1).strip())
        sid = f"writer.{_KIN_SYNONYMS.get(kin, kin)}"
        if sid in ids:
            return sid
    return None


def _clean_value(value: str) -> str:
    value = " ".join(value.split())
    value = value.strip("\"'\u201c\u201d\u2018\u2019` ")
    return value.rstrip(".,;:").strip()


def normalise_facts(facts: list[FactTuple], subjects: list[Subject]) -> list[FactTuple]:
    lookup = _subject_lookup(subjects)
    ids = {s.id for s in subjects}
    out: list[FactTuple] = []
    index: dict[tuple[str, str, str], int] = {}
    for fact in facts:
        subject = _resolve_subject(fact.subject, lookup, ids)
        value = _clean_value(fact.value)
        if subject is None or not value:
            continue
        relation = normalise_relation(fact.relation)
        key = (subject, relation, _ARTICLE.sub("", _key(value)))
        if key in index:
            kept = out[index[key]]
            if _RANK[fact.importance] > _RANK[kept.importance]:
                out[index[key]] = kept.model_copy(update={"importance": fact.importance})
            continue
        index[key] = len(out)
        out.append(
            fact.model_copy(update={"subject": subject, "relation": relation, "value": value})
        )
    return out
