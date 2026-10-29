"""Agent 1 orchestration: ``rule_detect -> infer_quasi -> merge_spans`` (spec 8.2).

OWNER: Person C (Platform and Integration).

``detect()`` is the only public entry point the CLI, the HTTP API and (in Phase
3) the full sanitisation pipeline call. It adds what none of the three stages
owns: timing, retry accounting, error-stage tagging, and validation at every
boundary between stages, so that a contract violation by any stage surfaces
here with a precise error instead of as a wrong score on Day 7.

The stage functions are injected through ``DetectorStages`` (defaulting to the
real modules) so each person's half can be tested with the other half faked.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from pydantic import ValidationError

from sanitise.agents.detector import merge, quasi, rules
from sanitise.config import get_settings
from sanitise.errors import (
    DocTooLong,
    OffsetOutOfRange,
    PipelineError,
    SchemaValidationFailed,
    Stage,
)
from sanitise.llm.base import LLMClient, format_validation_error, track_calls
from sanitise.llm.factory import detector_llm
from sanitise.loader import sha256_hex
from sanitise.rules import OffsetOutOfRangeError as RulesOffsetError
from sanitise.schemas import DetectorMeta, DetectorOutput, DetectorResult, SensitiveSpan
from sanitise.validation import check_span_offsets

RuleDetectFn = Callable[[str], list[SensitiveSpan]]
InferQuasiFn = Callable[[str, list[SensitiveSpan], LLMClient], Awaitable[list[SensitiveSpan]]]
MergeFn = Callable[[list[SensitiveSpan], list[SensitiveSpan]], list[SensitiveSpan]]


@dataclass(frozen=True)
class DetectorStages:
    rule_detect: RuleDetectFn
    infer_quasi: InferQuasiFn
    merge_spans: MergeFn


def default_stages() -> DetectorStages:
    # Looked up at call time so monkeypatching the modules in tests takes effect.
    return DetectorStages(
        rule_detect=rules.rule_detect,
        infer_quasi=quasi.infer_quasi,
        merge_spans=merge.merge_spans,
    )


@dataclass
class DetectorTrace:
    """Per-stage detail for ``--verbose`` output. Not part of the wire contract."""

    rule_spans: list[SensitiveSpan] = field(default_factory=list)
    llm_spans: list[SensitiveSpan] = field(default_factory=list)
    merged_spans: list[SensitiveSpan] = field(default_factory=list)
    rule_ms: int = 0
    llm_ms: int = 0
    merge_ms: int = 0
    total_ms: int = 0
    llm_calls: int = 0
    retries: int = 0
    llm_ran: bool = True


def _ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def _revalidate(spans: object, stage: Stage, who: str) -> list[SensitiveSpan]:
    """Re-run Pydantic validation on a stage's output (it may have been built with
    ``model_construct`` or mutated after construction)."""
    if not isinstance(spans, list):
        raise SchemaValidationFailed(
            f"{who} returned {type(spans).__name__}, expected a list", stage=stage
        )
    try:
        return [
            SensitiveSpan.model_validate(
                s.model_dump(warnings=False) if hasattr(s, "model_dump") else s
            )
            for s in spans
        ]
    except ValidationError as exc:
        raise SchemaValidationFailed(
            f"{who}: {format_validation_error(exc)}", stage=stage
        ) from None


def _check_doc(text: str, max_chars: int | None) -> None:
    limit = get_settings().max_doc_chars if max_chars is None else max_chars
    if len(text) > limit:
        raise DocTooLong(f"document has {len(text)} characters; the limit is {limit}", stage="load")


async def run_rule_pass(text: str, stages: DetectorStages) -> list[SensitiveSpan]:
    try:
        spans = await asyncio.to_thread(stages.rule_detect, text)
    except PipelineError as exc:
        raise exc.with_stage("rule_pass") from None
    except RulesOffsetError as exc:
        raise OffsetOutOfRange(f"rule_detect: {exc}", stage="rule_pass") from None
    except ValidationError as exc:
        raise SchemaValidationFailed(
            f"rule_detect: {format_validation_error(exc)}", stage="rule_pass"
        ) from None
    spans = _revalidate(spans, "rule_pass", "rule_detect")
    for i, span in enumerate(spans):
        if span.method != "rule":
            raise SchemaValidationFailed(
                f"rule_detect span {i} has method={span.method!r}, expected 'rule'",
                stage="rule_pass",
            )
        if span.start is None:
            raise OffsetOutOfRange(
                f"rule_detect span {i} {span.text!r} has no offsets; "
                "rule spans need exact start/end",
                stage="rule_pass",
            )
    check_span_offsets(text, spans, stage="rule_pass")
    return spans


async def run_llm_pass(
    text: str, explicit: list[SensitiveSpan], llm: LLMClient, stages: DetectorStages
) -> list[SensitiveSpan]:
    try:
        spans = await stages.infer_quasi(text, explicit, llm)
    except PipelineError as exc:
        raise exc.with_stage("llm_pass") from None
    except ValidationError as exc:
        raise SchemaValidationFailed(
            f"infer_quasi: {format_validation_error(exc)}", stage="llm_pass"
        ) from None
    spans = _revalidate(spans, "llm_pass", "infer_quasi")
    for i, span in enumerate(spans):
        if span.method != "llm":
            raise SchemaValidationFailed(
                f"infer_quasi span {i} has method={span.method!r}, expected 'llm'", stage="llm_pass"
            )
    check_span_offsets(text, spans, stage="llm_pass")
    return spans


def _model_name(llm: LLMClient) -> str:
    return str(getattr(llm, "model", type(llm).__name__))


async def detect_with_trace(
    doc_id: str,
    text: str,
    llm: LLMClient,
    *,
    run_llm: bool = True,
    stages: DetectorStages | None = None,
    max_chars: int | None = None,
) -> tuple[DetectorResult, DetectorTrace]:
    """``detect()`` plus the per-stage trace used by the verbose CLI output."""
    stages = stages or default_stages()
    trace = DetectorTrace(llm_ran=run_llm)
    t_total = time.perf_counter()
    _check_doc(text, max_chars)

    t = time.perf_counter()
    trace.rule_spans = await run_rule_pass(text, stages)
    trace.rule_ms = _ms(t)

    if run_llm:
        llm = detector_llm(llm)
        t = time.perf_counter()
        with track_calls() as stats:
            trace.llm_spans = await run_llm_pass(text, trace.rule_spans, llm, stages)
        trace.llm_ms = _ms(t)
        trace.llm_calls, trace.retries = stats.calls, stats.retries

    t = time.perf_counter()
    try:
        merged = stages.merge_spans(list(trace.rule_spans), list(trace.llm_spans))
    except PipelineError as exc:
        raise exc.with_stage("merge") from None
    except RulesOffsetError as exc:
        raise OffsetOutOfRange(f"merge_spans: {exc}", stage="merge") from None
    except ValidationError as exc:
        raise SchemaValidationFailed(
            f"merge_spans: {format_validation_error(exc)}", stage="merge"
        ) from None
    trace.merged_spans = _revalidate(merged, "merge", "merge_spans")
    check_span_offsets(text, trace.merged_spans, stage="merge")
    trace.merge_ms = _ms(t)
    trace.total_ms = _ms(t_total)

    result = DetectorResult(
        doc_id=doc_id,
        original_sha256=sha256_hex(text),
        output=DetectorOutput(doc_id=doc_id, spans=trace.merged_spans),
        meta=DetectorMeta(
            model=_model_name(llm) if run_llm else "none",
            rule_ms=trace.rule_ms,
            llm_ms=trace.llm_ms,
            total_ms=trace.total_ms,
            retries=trace.retries,
            rule_span_count=len(trace.rule_spans),
            llm_span_count=len(trace.llm_spans),
        ),
    )
    return result, trace


async def detect(doc_id: str, text: str, llm: LLMClient) -> DetectorResult:
    """rule_detect -> infer_quasi -> merge_spans, with timing and meta."""
    result, _ = await detect_with_trace(doc_id, text, llm)
    return result


async def detect_rules(
    doc_id: str, text: str, *, stages: DetectorStages | None = None, max_chars: int | None = None
) -> DetectorOutput:
    """Rule pass only (POST /v1/detect/rules) - lets Person A test without a model."""
    _check_doc(text, max_chars)
    spans = await run_rule_pass(text, stages or default_stages())
    return DetectorOutput(doc_id=doc_id, spans=spans)


async def detect_llm(
    doc_id: str,
    text: str,
    explicit: list[SensitiveSpan],
    llm: LLMClient,
    *,
    stages: DetectorStages | None = None,
    max_chars: int | None = None,
) -> DetectorOutput:
    """LLM pass only (POST /v1/detect/llm) - lets Person B test in isolation."""
    _check_doc(text, max_chars)
    explicit = _revalidate(explicit, "validation", "explicit[]")
    check_span_offsets(text, explicit, stage="validation")
    spans = await run_llm_pass(text, explicit, llm, stages or default_stages())
    return DetectorOutput(doc_id=doc_id, spans=spans)
