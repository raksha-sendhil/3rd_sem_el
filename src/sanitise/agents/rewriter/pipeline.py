"""Agent 3 orchestration.  OWNER: Person C (Platform and Integration).

    route (A) -> allocate_placeholders (A) -> generalise (A) -> reconstruct (B)
      -> leak_check (A) -> on a leak: reconstruct once more with feedback (B)
      -> still leaking: mask_fallback (A) -> still leaking: LEAK_CHECK_FAILED

``rewrite`` takes no raw text (spec 10.2): the agent sees the detected spans and the facts,
never the original wording. ``apply_decisions`` is the deterministic rewrite mock mode uses
instead of the model: the original text with every placeholder and generalised span
substituted in place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from sanitise.agents.rewriter import leakcheck, ontology, placeholders, prompt, routing
from sanitise.errors import LeakCheckFailed, PipelineError, SchemaValidationFailed
from sanitise.llm.base import LLMClient, format_validation_error
from sanitise.schemas import (
    Action,
    DetectorOutput,
    ExtractorOutput,
    RewriterOutput,
    RoutingDecision,
    SensitiveSpan,
)


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
        before = _fit(out[: d.start], d.replacement)
        out = before + d.replacement + out[d.end :]
        last_start = len(before)
    return out


# A generalised phrase brings its own article ("a magistrate's court"), so the word that
# introduced the original goes: "before the Pune District Court" -> "before a district court",
# "M/s Kale Traders" -> "a trading business". A date that lost its day reads "in", not "on".
_LEADING_ARTICLE = re.compile(r"(?:\b(?:the|a|an)|\bM/[sS]\.?)\s+$", re.IGNORECASE)
_STARTS_WITH_ARTICLE = re.compile(r"(?:a|an|the)\b", re.IGNORECASE)
_NO_DAY = re.compile(r"^(?:[A-Z][a-z]+ )?(?:19|20)\d{2}$|^the (?:early |mid |late )?\d{4}s$")
_ON = re.compile(r"\bon(\s+)$")


def _fit(before: str, replacement: str) -> str:
    if _STARTS_WITH_ARTICLE.match(replacement):
        trimmed = _LEADING_ARTICLE.sub("", before)
        if trimmed != before:
            return trimmed
    if _NO_DAY.match(replacement) and _ON.search(before):
        return _ON.sub(r"in\1", before)
    return before


def removed_decisions(decisions: list[RoutingDecision]) -> list[RoutingDecision]:
    """What the leak check looks for: positional spans that were replaced."""
    return [d for d in decisions if d.action is not Action.KEEP and d.start is not None]


def rewrite_in_place(
    doc_id: str, text: str, detected: DetectorOutput, extracted: ExtractorOutput
) -> RewriteTrace:
    """Placeholders and ladders applied directly to the original text, with the leak check and
    mask (the spec's fallback when the rewrite drops the advice, and mock mode's rewrite)."""
    decisions, tokens = plan_decisions(detected, extracted)
    removed = removed_decisions(decisions)
    safe = apply_decisions(text, decisions)
    leaks = leakcheck.leak_check(removed, safe)
    if leaks:
        safe = leakcheck.mask_fallback(safe, leaks, tokens)
        leaks = leakcheck.leak_check(removed, safe)
    output = RewriterOutput(doc_id=doc_id, safe_text=safe, decisions=decisions, placeholders=tokens)
    return RewriteTrace(output=output, residual_leaks=leaks)


async def rewrite_with_trace(
    doc_id: str,
    detected: DetectorOutput,
    extracted: ExtractorOutput,
    llm: LLMClient,
) -> RewriteTrace:
    try:
        decisions, tokens = plan_decisions(detected, extracted)
        removed = removed_decisions(decisions)
        safe = await prompt.reconstruct(extracted.facts, decisions, tokens, llm)
        leaks = leakcheck.leak_check(removed, safe)
        retries = 0
        if leaks:
            retries = 1
            safe = await prompt.reconstruct(extracted.facts, decisions, tokens, llm, feedback=leaks)
            leaks = leakcheck.leak_check(removed, safe)
        if leaks:
            safe = leakcheck.mask_fallback(safe, leaks, tokens)
            leaks = leakcheck.leak_check(removed, safe)
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
