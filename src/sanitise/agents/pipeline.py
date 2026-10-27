"""The offline sanitisation path: Agent 1 -> Agent 2 -> Agent 3.  OWNER: Person C.

``sanitise(doc_id, text, llm)`` is the one entry point the batch runner, the CLI and the
HTTP API use (docs/phase3_plan.md 6.2).
"""

from __future__ import annotations

import time

from sanitise.agents.detector.pipeline import detect
from sanitise.agents.extractor.pipeline import extract
from sanitise.agents.rewriter.pipeline import rewrite_with_trace
from sanitise.llm.base import LLMClient, track_calls
from sanitise.loader import sha256_hex
from sanitise.schemas import SanitisedRecord, SanitiseMeta


def _ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


async def sanitise(doc_id: str, text: str, llm: LLMClient) -> SanitisedRecord:
    t_total = time.perf_counter()
    with track_calls() as stats:
        t = time.perf_counter()
        detected = await detect(doc_id, text, llm)
        detect_ms = _ms(t)
        t = time.perf_counter()
        subs, extracted = await extract(doc_id, text, detected.output, llm)
        extract_ms = _ms(t)
        t = time.perf_counter()
        trace = await rewrite_with_trace(doc_id, detected.output, extracted, llm)
        rewrite_ms = _ms(t)
    return SanitisedRecord(
        doc_id=doc_id,
        original_sha256=sha256_hex(text),
        subjects=subs,
        detector=detected.output,
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
        ),
    )
