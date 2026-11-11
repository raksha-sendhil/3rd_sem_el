"""The offline sanitisation path: Agent 1 -> Agent 2 -> Agent 3.  OWNER: Person C.

``sanitise(doc_id, text, llm)`` is the one entry point the batch runner, the CLI and the
HTTP API use (docs/phase3_plan.md 6.2).

In the legal domain, broken model output does not lose the document (legal build spec, "Risks
and fallbacks"): Agent 1 falls back to its rules, Agent 2 to one fact per detected span, and
Agent 3 to placeholders and ladders applied in place, also when its rewrite drops the advice.
``meta.fallback`` records each one and the report counts them. Only broken output falls back:
an unreachable or timed-out model still fails the document, so a re-run can process it properly.
"""

from __future__ import annotations

import re
import time

from sanitise_legal.agents.detector.pipeline import detect, detect_rules
from sanitise_legal.agents.extractor import canonical
from sanitise_legal.agents.extractor.pipeline import extract, rules_only_facts
from sanitise_legal.agents.rewriter.pipeline import rewrite_in_place, rewrite_with_trace
from sanitise_legal.domain import is_legal
from sanitise_legal.errors import LeakCheckFailed, SchemaValidationFailed
from sanitise_legal.llm.base import LLMClient, track_calls
from sanitise_legal.loader import sha256_hex
from sanitise_legal.schemas import FactTuple, SanitisedRecord, SanitiseMeta


def _ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


_WORD = re.compile(r"[a-z0-9]+")


def _keeps_advice(facts: list[FactTuple], safe_text: str) -> bool:
    """Is most of each piece of advice still in the rewrite? A paraphrase passes; a rewrite that
    dropped it does not. The advice is what lawyers search for."""
    said = set(_WORD.findall(safe_text.lower()))
    for fact in facts:
        if fact.relation == "advised":
            words = set(_WORD.findall(fact.value.lower()))
            if words and len(words & said) / len(words) < 0.6:
                return False
    return True


async def sanitise(
    doc_id: str, text: str, llm: LLMClient, *, detector_llm: LLMClient | None = None
) -> SanitisedRecord:
    """``detector_llm`` runs Agent 1's LLM pass when given (the fine-tuned Gemma 4 on its own
    GPU, legal build spec Part 3); Agents 2 and 3 always use ``llm``."""
    t_total = time.perf_counter()
    with track_calls() as stats:
        legal = is_legal()
        fallback: list[str] = []
        t = time.perf_counter()
        try:
            detector = (await detect(doc_id, text, detector_llm or llm)).output
        except SchemaValidationFailed:
            if not legal:
                raise
            detector = await detect_rules(doc_id, text)
            fallback.append("agent1_rules_only")
        detect_ms = _ms(t)
        t = time.perf_counter()
        try:
            subs, extracted = await extract(doc_id, text, detector, llm)
        except SchemaValidationFailed:
            if not legal:
                raise
            subs = canonical.subjects(text, detector.spans)
            extracted = rules_only_facts(doc_id, detector)
            fallback.append("agent2_rules_only")
        extract_ms = _ms(t)
        t = time.perf_counter()
        if "agent2_rules_only" in fallback:  # the facts hold no advice: keep the original wording
            trace = rewrite_in_place(doc_id, text, detector, extracted)
            fallback.append("agent3_in_place")
        else:
            try:
                trace = await rewrite_with_trace(doc_id, detector, extracted, llm)
                usable = (
                    bool(extracted.facts)
                    and bool(trace.output.safe_text.strip())
                    and _keeps_advice(extracted.facts, trace.output.safe_text)
                )
            except (SchemaValidationFailed, LeakCheckFailed):
                if not legal:
                    raise
                usable = False
            if legal and not usable:
                trace = rewrite_in_place(doc_id, text, detector, extracted)
                fallback.append("agent3_in_place")
        rewrite_ms = _ms(t)
    return SanitisedRecord(
        doc_id=doc_id,
        original_sha256=sha256_hex(text),
        subjects=subs,
        detector=detector,
        extractor=extracted,
        rewriter=trace.output,
        residual_leaks=trace.residual_leaks,
        meta=SanitiseMeta(
            model=str(getattr(llm, "model", type(llm).__name__)),
            detect_ms=detect_ms,
            extract_ms=extract_ms,
            rewrite_ms=rewrite_ms,
            total_ms=_ms(t_total),
            retries=stats.retries,
            leak_retries=trace.leak_retries,
            fallback=fallback,
        ),
    )
