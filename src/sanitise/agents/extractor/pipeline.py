"""Agent 2 orchestration.  OWNER: Person C (Platform and Integration).

``extract`` runs Person A's ``subjects``, the ``extract_facts`` prompt (one call per subject)
and Person A's ``normalise_facts`` in that order and re-validates the boundary between each
step, so a contract violation surfaces here with the stage tagged ``extract``.

Two parts of the result never come from the model:

* ``identity_facts``: names, ID numbers, contact details and kinship, built from Agent 1's
  spans and the subjects. A name belongs to the subject it is a mention of; an ID number,
  phone or email to the person mentioned last before it in the same sentence (the writer if
  no one is); ``writer.<relative>`` gets ``relation_to: <relative>``.
* ``in_record``: a model fact is kept only if its value is worded from the record (at least
  half of its words of three letters or more appear in the text), so a detail the model
  invented is dropped before Agent 3 can write it out.
"""

from __future__ import annotations

import logging
import re

from pydantic import ValidationError

from sanitise.agents.extractor import canonical, importance, prompt
from sanitise.errors import PipelineError, SchemaValidationFailed
from sanitise.llm.base import LLMClient, format_validation_error
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


def _fact(subject: str, relation: str, value: str) -> FactTuple:
    return FactTuple(
        subject=subject,
        relation=relation,
        value=value,
        importance=importance.default_importance(relation),
    )


def _owner(text: str, span: SensitiveSpan, subjects: list[Subject]) -> str:
    """The subject mentioned last before ``span`` in its sentence (longest mention on a tie)."""
    assert span.start is not None
    if span.category is C.NAME:
        named = next((s.id for s in subjects if span.text in s.mentions), None)
        if named:
            return named
    sentence = max((m.end() for m in _SENTENCE_END.finditer(text, 0, span.start)), default=0)
    best: tuple[int, int, str] | None = None
    for s in subjects:
        for mention in s.mentions:
            pattern = rf"(?<!\w){re.escape(mention)}(?!\w)"
            for m in re.finditer(pattern, text[sentence : span.start]):
                key = (m.start(), len(mention), s.id)
                if best is None or key[:2] > best[:2]:
                    best = key
    return best[2] if best else "writer"


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


async def extract(
    doc_id: str, text: str, detected: DetectorOutput, llm: LLMClient
) -> tuple[list[Subject], ExtractorOutput]:
    try:
        subs = [
            Subject.model_validate(s.model_dump()) for s in canonical.subjects(text, detected.spans)
        ]
        raw = await prompt.extract_facts(text, subs, detected.spans, llm)
        grounded = [f for f in raw if in_record(f.value, text)]
        if len(grounded) < len(raw):
            log.info(
                "%s: dropped %d facts not worded from the record", doc_id, len(raw) - len(grounded)
            )
        facts = [
            FactTuple.model_validate(f.model_dump())
            for f in canonical.normalise_facts(
                identity_facts(text, subs, detected.spans) + grounded, subs
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
