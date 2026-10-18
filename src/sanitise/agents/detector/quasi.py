"""Stage 4 - LLM quasi-identifier pass.  OWNER: Person B (Model and Prompt).

PLACEHOLDER created by Person C: a deliberately minimal prompt so the wiring
can be exercised against MockLLM and Ollama on Day 1. Replace the prompt text
and add few-shot examples; keep the signature and keep calling the model only
through ``llm.json_call`` (never import an Ollama client here).

What the platform already provides through ``llm.json_call(system, user, QuasiList)``:

* the ``QuasiList`` JSON schema is sent to Ollama as ``format`` (constrained decoding);
* output is parsed and validated against ``QuasiList``; on failure it retries up
  to ``retries`` times with escalating temperature, feeding the validation error
  back as a repair instruction (see ``sanitise.llm.base``);
* after the last failure it raises ``SchemaValidationFailed`` (stage ``llm_pass``).

To change the repair wording or the lenient parser, pass ``repair_builder=`` /
``parser=`` hooks to the client (see ``sanitise.llm.factory``) rather than
re-implementing the loop.
"""

from __future__ import annotations

import json

from sanitise.llm.base import LLMClient
from sanitise.schemas import QuasiList, SensitiveSpan

# TODO(Person B): real system prompt + few-shot examples.
SYSTEM_PROMPT = (
    "You find quasi-identifiers in a document: combinations of details that could "
    "re-identify a person even though no single detail is an explicit identifier "
    "(for example a rare occupation together with a specific condition, or a precise "
    "timeline together with a small location). Explicit identifiers are already "
    "listed; do not repeat them. Return only quasi-identifiers you can justify, each "
    "with a short rationale. Return an empty list if there are none."
)


def build_user_prompt(text: str, explicit: list[SensitiveSpan]) -> str:
    found = [{"text": s.text, "category": s.category.value} for s in explicit]
    return f"DOCUMENT:\n{text}\n\nEXPLICIT IDENTIFIERS ALREADY FOUND:\n{json.dumps(found)}"


async def infer_quasi(
    text: str, explicit: list[SensitiveSpan], llm: LLMClient
) -> list[SensitiveSpan]:
    """One constrained-JSON call. Returns spans with method='llm', start/end=None."""
    result = await llm.json_call(SYSTEM_PROMPT, build_user_prompt(text, explicit), QuasiList)
    return [
        SensitiveSpan(
            text=hit.text,
            category=hit.category,
            method="llm",
            rationale=hit.rationale,
        )
        for hit in result.quasi_identifiers
    ]
