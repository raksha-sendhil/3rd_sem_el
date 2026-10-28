"""Agent 2 orchestration.  OWNER: Person C (Platform and Integration).

``extract`` runs Person A's ``subjects``, the ``extract_facts`` prompt (one call per subject)
and Person A's ``normalise_facts`` in that order and re-validates the boundary between each
step, so a contract violation surfaces here with the stage tagged ``extract``.

Two parts of the result never come from the model:

* ``identity_facts``: names, ID numbers, contact details and kinship, built from Agent 1's
  spans and the subjects. A name belongs to the subject it is a mention of; an ID number,
  phone or email to the person mentioned last before it in the same sentence (the writer if
  no one is); ``writer.<relative>`` gets ``relation_to: <relative>``.
* ``check_facts``: Agent 2 may only describe what Agent 1 found. An age, place, job, date,
  symptom or condition the model lists must match a detail Agent 1 detected, of the right
  kind; it takes Agent 1's wording and goes to the person mentioned last before it in its
  sentence ("My son ... has abdominal pain" -> the son). A fact that matches nothing is
  dropped, so the model cannot add a detail ("sunlight" as a symptom, an email domain as a
  home) or give one person's detail to another. Medication and procedures, which Agent 1 does
  not detect, must be worded from the record; ``other`` is never kept.
"""

from __future__ import annotations

import logging
import re

from pydantic import ValidationError

from sanitise.agents.extractor import canonical, importance, prompt
from sanitise.errors import PipelineError, SchemaValidationFailed
from sanitise.llm.base import LLMClient, format_validation_error
from sanitise.llm.factory import agents_llm
from sanitise.schemas import (
    DetectorOutput,
    ExtractorOutput,
    FactTuple,
    PIICategory,
    SensitiveSpan,
    Subject,
)

log = logging.getLogger(__name__)

C = PIICategory
OWNED = {C.NAME: "identifier", C.ID_NUMBER: "identifier", C.PHONE: "contact", C.EMAIL: "contact"}
_WORD = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_END = re.compile(r"[.!?\n]")
# relation -> the Agent 1 categories a fact of that relation must match
SPAN_RELATIONS: dict[str, set[PIICategory]] = {
    "age": {C.AGE},
    "height": {C.PHYSICAL},
    "weight": {C.PHYSICAL},
    "lives_in": {C.LOCATION, C.ADDRESS},
    "studies_at": {C.LOCATION, C.ADDRESS},
    "treated_at": {C.LOCATION, C.ADDRESS},
    "works_as": {C.OCCUPATION},
    "event_date": {C.DATE},
    "has_condition": {C.MEDICAL_CONDITION, C.SYMPTOM},
    "has_symptom": {C.SYMPTOM, C.MEDICAL_CONDITION},
}
_COUNT = (
    r"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|few"
    r"|several|couple\s+of)"
)
_DURATION = re.compile(
    rf"\b{_COUNT}\s+(?:minute|hour|day|night|week|month|year)s?\b", re.IGNORECASE
)
_SEX = re.compile(r"male|female|man|woman|boy|girl", re.IGNORECASE)


def _fact(subject: str, relation: str, value: str) -> FactTuple:
    return FactTuple(
        subject=subject,
        relation=relation,
        value=value,
        importance=importance.default_importance(relation),
    )


def _subject_at(text: str, pos: int, subjects: list[Subject]) -> str | None:
    """The subject mentioned last before ``pos`` in its sentence (longest mention on a tie)."""
    sentence = max((m.end() for m in _SENTENCE_END.finditer(text, 0, pos)), default=0)
    best: tuple[int, int, str] | None = None
    for s in subjects:
        for mention in s.mentions:
            pattern = rf"(?<!\w){re.escape(mention)}(?!\w)"
            for m in re.finditer(pattern, text[sentence:pos]):
                key = (m.start(), len(mention), s.id)
                if best is None or key[:2] > best[:2]:
                    best = key
    return best[2] if best else None


def _owner(text: str, span: SensitiveSpan, subjects: list[Subject]) -> str:
    assert span.start is not None
    if span.category is C.NAME:
        named = next((s.id for s in subjects if span.text in s.mentions), None)
        if named:
            return named
    return _subject_at(text, span.start, subjects) or "writer"


def identity_facts(
    text: str, subjects: list[Subject], spans: list[SensitiveSpan]
) -> list[FactTuple]:
    facts = [
        _fact(s.id, "relation_to", s.id.removeprefix("writer.").replace("_", " "))
        for s in subjects
        if s.id.startswith("writer.")
    ]
    for span in sorted(
        (s for s in spans if s.category in OWNED and s.start is not None),
        key=lambda s: s.start or 0,
    ):
        facts.append(_fact(_owner(text, span, subjects), OWNED[span.category], span.text))
    return facts


def in_record(value: str, text: str) -> bool:
    words = [w.lower() for w in _WORD.findall(value) if len(w) >= 3 or w.isdigit()]
    if not words:
        return True
    present = {w.lower() for w in _WORD.findall(text)}
    return sum(w in present for w in words) * 2 >= len(words)


def _within(short: str, long: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(short.strip())}(?!\w)", long, re.IGNORECASE) is not None


def _words(value: str) -> str:
    return " ".join(w.lower() for w in _WORD.findall(value))


def check_facts(
    text: str,
    facts: list[FactTuple],
    spans: list[SensitiveSpan],
    subjects: list[Subject],
    *,
    trust_subjects: bool = False,
) -> list[FactTuple]:
    """The model's facts that describe something in the record, each given to its person.

    * A relation in ``SPAN_RELATIONS`` must name a detail Agent 1 found, of a matching category
      (the value contains it or is part of it). The fact takes the detected text as its value;
      one fact per detail. Anything else is dropped: "sunlight" as a symptom, an email domain
      as a home.
    * ``symptom_duration`` must be a length of time and ``sex`` a sex. The rest (medication,
      procedures, ``other``) must be worded from the record.
    * Of two facts with the same subject and relation, one whose words are contained in the
      other's ("breathless", "breathless climbing stairs") is dropped.

    Who a fact is about: with ``trust_subjects`` (a strong model that saw every person at once
    and chose from their ids), the model's subject. Otherwise (one call per person, where a
    small model lists the whole record for everyone) the subject mentioned last before the
    detail in its sentence, and ``other`` is dropped (it was mostly noise). The rule that an
    age, place, job, date, symptom or condition must be one Agent 1 found holds for every model.
    """
    placed = []
    lower = text.lower()
    for f in facts:
        categories = SPAN_RELATIONS.get(f.relation)
        if categories is not None:
            for s in spans:
                if (
                    s.category in categories
                    and s.start is not None
                    and (_within(s.text, f.value) or _within(f.value, s.text))
                ):
                    owner = f.subject if trust_subjects else _subject_at(text, s.start, subjects)
                    placed.append(_fact(owner or f.subject, f.relation, s.text))
            continue
        if (f.relation == "other" and not trust_subjects) or not in_record(f.value, text):
            continue
        if f.relation == "symptom_duration" and not _DURATION.search(f.value):
            continue
        if f.relation == "sex" and not _SEX.fullmatch(f.value.strip()):
            continue
        at = lower.find(f.value.lower())
        owner = _subject_at(text, at, subjects) if at >= 0 and not trust_subjects else None
        placed.append(f.model_copy(update={"subject": owner or f.subject}))
    keys = [(f.subject, f.relation, f" {_words(f.value)} ") for f in placed]
    return [
        f
        for f, (subject, relation, words) in zip(placed, keys, strict=True)
        if not any(
            (s, r) == (subject, relation) and other != words and words in other
            for s, r, other in keys
        )
    ]


async def extract(
    doc_id: str, text: str, detected: DetectorOutput, llm: LLMClient
) -> tuple[list[Subject], ExtractorOutput]:
    try:
        subs = [
            Subject.model_validate(s.model_dump()) for s in canonical.subjects(text, detected.spans)
        ]
        model = agents_llm(llm)
        raw = await prompt.extract_facts(text, subs, detected.spans, model)
        checked = check_facts(
            text,
            raw,
            detected.spans,
            subs,
            trust_subjects=getattr(model, "one_call_per_document", False),
        )
        log.info("%s: kept %d of the model's %d facts", doc_id, len(checked), len(raw))
        facts = [
            FactTuple.model_validate(f.model_dump())
            for f in canonical.normalise_facts(
                identity_facts(text, subs, detected.spans) + checked, subs
            )
        ]
    except PipelineError as exc:
        raise exc.with_stage("extract") from None
    except ValidationError as exc:
        raise SchemaValidationFailed(
            f"extract: {format_validation_error(exc)}", stage="extract"
        ) from None
    known = {s.id for s in subs}
    kept = [f for f in facts if f.subject in known]
    if len(kept) < len(facts):
        log.info("%s: dropped %d facts with unknown subjects", doc_id, len(facts) - len(kept))
    return subs, ExtractorOutput(doc_id=doc_id, facts=kept)
