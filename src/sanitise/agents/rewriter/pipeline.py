"""Agent 3 orchestration.  OWNER: Person C (Platform and Integration).

    route (A) -> allocate_placeholders (A) -> generalise (A) -> reconstruct (C)
      -> leak_check (A) + grounding check (C)
      -> on a problem: reconstruct once more with feedback (C)
      -> still a problem: mask_fallback (A), then grounding.mask (C)
      -> still a problem: LEAK_CHECK_FAILED

The leak check finds the removed details in any form; the grounding check finds names,
numbers and email addresses that are not in the facts the model was given at all.

``rewrite`` takes no raw text (spec 10.2): the agent sees the detected spans and the facts,
never the original wording. ``apply_decisions`` is the deterministic rewrite mock mode uses
instead of the model: the original text with every placeholder and generalised span
substituted in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from sanitise.agents.rewriter import (
    grounding,
    leakcheck,
    ontology,
    placeholders,
    prompt,
    routing,
)
from sanitise.casefile import case_file
from sanitise.errors import LeakCheckFailed, PipelineError, SchemaValidationFailed
from sanitise.llm.base import LLMClient, format_validation_error
from sanitise.llm.factory import agents_llm
from sanitise.schemas import (
    Action,
    DetectorOutput,
    ExtractorOutput,
    RewriterOutput,
    RoutingDecision,
    SensitiveSpan,
)

NO_FACTS = "The record states nothing that can be shared."


@dataclass
class RewriteTrace:
    output: RewriterOutput
    leak_retries: int = 0
    residual_leaks: list[SensitiveSpan] = field(default_factory=list)


def plan_decisions(
    detected: DetectorOutput, extracted: ExtractorOutput
) -> tuple[list[RoutingDecision], dict[str, str]]:
    """Routing decisions with every replacement filled in, plus the placeholder map."""
    decisions = [
        RoutingDecision.model_validate(d.model_dump())
        for d in routing.route(detected.spans, extracted.facts)
    ]
    if len(decisions) != len(detected.spans):
        raise SchemaValidationFailed(
            f"route returned {len(decisions)} decisions for {len(detected.spans)} spans",
            stage="rewrite",
        )
    tokens = placeholders.allocate_placeholders(decisions)
    for d in decisions:
        if d.action is Action.GENERALISE and d.replacement is None:
            d.replacement = ontology.generalise(d.span_text, d.category)
    return decisions, tokens


def apply_decisions(text: str, decisions: list[RoutingDecision]) -> str:
    """Substitute every positional placeholder / generalise decision into ``text``."""
    edits = sorted(
        (
            d
            for d in decisions
            if d.action is not Action.KEEP
            and d.start is not None
            and d.end is not None
            and d.replacement
        ),
        key=lambda d: d.start or 0,
        reverse=True,
    )
    out, last_start = text, len(text) + 1
    for d in edits:
        assert d.start is not None and d.end is not None and d.replacement is not None
        if d.end > last_start:  # overlaps an edit already applied
            continue
        out = out[: d.start] + d.replacement + out[d.end :]
        last_start = d.start
    return out


def removed_decisions(decisions: list[RoutingDecision]) -> list[RoutingDecision]:
    """What the leak check looks for: positional spans that were replaced."""
    return [d for d in decisions if d.action is not Action.KEEP and d.start is not None]


async def rewrite_with_trace(
    doc_id: str,
    detected: DetectorOutput,
    extracted: ExtractorOutput,
    llm: LLMClient,
) -> RewriteTrace:
    llm = agents_llm(llm)
    with case_file(prompt.is_case_file(extracted.facts)):
        return await _rewrite(doc_id, detected, extracted, llm)


async def _rewrite(
    doc_id: str, detected: DetectorOutput, extracted: ExtractorOutput, llm: LLMClient
) -> RewriteTrace:
    try:
        decisions, tokens = plan_decisions(detected, extracted)
        removed = removed_decisions(decisions)
        facts_text = prompt.draft(extracted.facts, decisions)

        def problems(text: str) -> list[SensitiveSpan]:
            return leakcheck.leak_check(removed, text) + grounding.ungrounded(text, facts_text)

        if not facts_text:  # nothing to write from: no model call, no made-up text
            safe = NO_FACTS
        else:
            safe = await prompt.reconstruct(extracted.facts, decisions, tokens, llm)
        leaks = problems(safe)
        retries = 0
        if leaks:
            retries = 1
            safe = await prompt.reconstruct(extracted.facts, decisions, tokens, llm, feedback=leaks)
            leaks = problems(safe)
        if leaks:
            safe = leakcheck.mask_fallback(safe, leakcheck.leak_check(removed, safe), tokens)
            safe = grounding.mask(safe, grounding.ungrounded(safe, facts_text))
            leaks = problems(safe)
        if leaks:
            raise LeakCheckFailed(
                f"{len(leaks)} removed details still present after retry and masking",
                stage="rewrite",
            )
        output = RewriterOutput(
            doc_id=doc_id, safe_text=safe, decisions=decisions, placeholders=tokens
        )
    except PipelineError as exc:
        raise exc.with_stage("rewrite") from None
    except ValidationError as exc:
        raise SchemaValidationFailed(
            f"rewrite: {format_validation_error(exc)}", stage="rewrite"
        ) from None
    return RewriteTrace(output=output, leak_retries=retries, residual_leaks=leaks)


async def rewrite(
    doc_id: str,
    detected: DetectorOutput,
    extracted: ExtractorOutput,
    llm: LLMClient,
) -> RewriterOutput:
    """No raw-text parameter, by design: Agent 3 rebuilds the record from facts alone."""
    return (await rewrite_with_trace(doc_id, detected, extracted, llm)).output
