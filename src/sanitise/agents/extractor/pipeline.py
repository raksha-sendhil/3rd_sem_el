"""Agent 2 orchestration.  OWNER: Person C (Platform and Integration).

``extract`` runs Person A's ``subjects``, the ``extract_facts`` prompt and Person A's
``normalise_facts`` in that order and re-validates the boundary between each step, so a
contract violation surfaces here with the stage tagged ``extract``.
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from sanitise.agents.extractor import canonical, prompt
from sanitise.errors import PipelineError, SchemaValidationFailed
from sanitise.llm.base import LLMClient, format_validation_error
from sanitise.schemas import DetectorOutput, ExtractorOutput, FactTuple, Subject

log = logging.getLogger(__name__)


async def extract(
    doc_id: str, text: str, detected: DetectorOutput, llm: LLMClient
) -> tuple[list[Subject], ExtractorOutput]:
    try:
        subs = [
            Subject.model_validate(s.model_dump()) for s in canonical.subjects(text, detected.spans)
        ]
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
