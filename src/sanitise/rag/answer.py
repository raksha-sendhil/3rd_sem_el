"""The online query path for both pipelines.  OWNER: Person C.

    question -> retriever.search (over ORIGINALS) -> top-k doc IDs (same for both pipelines)
      baseline:  original texts  -> answer prompt (B) -> LLM -> answer
      protected: sanitised texts -> answer prompt (B) -> LLM -> monitor hook -> answer

``RagContext`` holds the store, retriever, monitor and sessions; the API builds one at
startup. ``payload`` in every ``RagAnswer`` is the literal user message the model received.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from sanitise.errors import IndexNotBuilt, PipelineError
from sanitise.llm.base import LLMClient
from sanitise.monitor import AllowAllMonitor, SessionMonitor
from sanitise.rag.answer_prompt import build_answer_prompt
from sanitise.rag.retrieve import Retriever
from sanitise.rag.sessions import SessionStore
from sanitise.rag.store import DocumentStore
from sanitise.schemas import (
    AttackRound,
    CompareResponse,
    MonitorVerdict,
    Pipeline,
    RagAnswer,
)


@dataclass
class RagContext:
    store: DocumentStore
    retriever: Retriever
    monitor: SessionMonitor = field(default_factory=AllowAllMonitor)
    sessions: SessionStore = field(default_factory=SessionStore)


def _ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


async def answer(
    ctx: RagContext,
    query: str,
    pipeline: Pipeline,
    session_id: str,
    llm: LLMClient,
    k: int = 4,
) -> RagAnswer:
    t = time.perf_counter()
    if not ctx.store.ids():
        raise IndexNotBuilt("the document store is empty", stage="retrieve")
    try:
        hits = ctx.retriever.search(query, k)
        if pipeline == "baseline":
            contexts = [(h.doc_id, ctx.store.original(h.doc_id)) for h in hits]
        else:
            contexts = [(h.doc_id, ctx.store.sanitised(h.doc_id).rewriter.safe_text) for h in hits]
        system, user = build_answer_prompt(query, contexts)
        text = await llm.text_call(system, user)
        verdict = (
            ctx.monitor.check(session_id, query, hits, text)
            if pipeline == "protected"
            else MonitorVerdict(reason="baseline pipeline has no monitor")
        )
    except PipelineError as exc:
        raise exc.with_stage("answer") from None
    if verdict.verdict == "blocked":
        text = "This question cannot be answered: it would narrow the records down too far."
    return RagAnswer(
        pipeline=pipeline,
        query=query,
        answer=text,
        doc_ids=[h.doc_id for h in hits],
        payload=user,
        monitor=verdict,
        latency_ms=_ms(t),
    )


async def compare(
    ctx: RagContext, query: str, session_id: str, llm: LLMClient, k: int = 4
) -> CompareResponse:
    """Both pipelines on the same question; the round is appended to the session trace."""
    baseline, protected = await asyncio.gather(
        answer(ctx, query, "baseline", session_id, llm, k),
        answer(ctx, query, "protected", session_id, llm, k),
    )
    ctx.sessions.append(
        session_id,
        AttackRound(
            round=ctx.sessions.next_round(session_id),
            query=query,
            baseline=baseline,
            protected=protected,
        ),
    )
    return CompareResponse(baseline=baseline, protected=protected)
