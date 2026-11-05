"""Agent 2 orchestration.  OWNER: Person C (Platform and Integration).

``extract`` runs Person A's ``subjects``, the ``extract_facts`` prompt and Person A's
``normalise_facts`` in that order and re-validates the boundary between each step, so a
contract violation surfaces here with the stage tagged ``extract``.
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from sanitise.agents.extractor import canonical, importance, prompt
from sanitise.domain import is_legal
from sanitise.errors import PipelineError, SchemaValidationFailed
from sanitise.llm.base import LLMClient, format_validation_error
from sanitise.schemas import DetectorOutput, ExtractorOutput, FactTuple, PIICategory, Subject

log = logging.getLogger(__name__)


C = PIICategory
_RULES_RELATION = {
    C.NAME: "identifier",
    C.ID_NUMBER: "identifier",
    C.CASE_NUMBER: "identifier",
    C.FINANCIAL: "identifier",
    C.MATTER_CODENAME: "identifier",
    C.PHONE: "contact",
    C.EMAIL: "contact",
    C.ADDRESS: "lives_in",
    C.LOCATION: "lives_in",
    C.AGE: "age",
    C.OCCUPATION: "works_as",
    C.ORGANIZATION: "opposing_party",
    C.COURT: "filed_in",
    C.MONEY: "claim_amount",
    C.DATE: "filed_on",
    C.LEGAL_ISSUE: "has_legal_issue",
}


def rules_only_facts(doc_id: str, detected: DetectorOutput) -> ExtractorOutput:
    """The legal fallback when Agent 2's model keeps writing broken JSON: one fact per detected
    span. It holds no advice or outcome, so these documents lower Agent 2's coverage score."""
    facts = [
        FactTuple(
            subject="client",
            relation=_RULES_RELATION[s.category],
            value=s.text,
            importance=importance.default_importance(_RULES_RELATION[s.category]),
        )
        for s in detected.spans
        if s.start is not None and s.category in _RULES_RELATION
    ]
    return ExtractorOutput(doc_id=doc_id, facts=facts)


async def extract(
    doc_id: str, text: str, detected: DetectorOutput, llm: LLMClient
) -> tuple[list[Subject], ExtractorOutput]:
    try:
        subs = [
            Subject.model_validate(s.model_dump()) for s in canonical.subjects(text, detected.spans)
        ]
        # A matter always has a client, even when no name was recognised as the client's: the
        # advice and the outcome are filed under this subject and would otherwise be dropped.
        if is_legal() and not any(s.id == "client" for s in subs):
            subs.append(Subject(id="client", label="the client"))
        raw = await prompt.extract_facts(text, subs, detected.spans, llm)
        facts = [
            FactTuple.model_validate(f.model_dump()) for f in canonical.normalise_facts(raw, subs)
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
