"""Session privacy monitor hook (docs/phase3_plan.md 6.7).  OWNER: Person C.

The protected pipeline calls ``monitor.check(...)`` before returning every answer.
``AllowAllMonitor`` always allows; ``CandidatePoolMonitor`` (Phase 3b, Person A, pool.py)
fills ``candidate_pool`` and ``disclosed`` and blocks when fewer than 5 clients would still
match. ``SANITISE_MONITOR`` picks one (``build_monitor``).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sanitise_legal.schemas import MonitorVerdict, RetrievalHit


@runtime_checkable
class SessionMonitor(Protocol):
    def check(
        self, session_id: str, query: str, hits: list[RetrievalHit], draft_answer: str
    ) -> MonitorVerdict: ...


class AllowAllMonitor:
    """The Phase 3a stub."""

    def check(
        self, session_id: str, query: str, hits: list[RetrievalHit], draft_answer: str
    ) -> MonitorVerdict:
        return MonitorVerdict()


from sanitise_legal.monitor.pool import CandidatePoolMonitor, build_monitor  # noqa: E402

__all__ = ["AllowAllMonitor", "CandidatePoolMonitor", "SessionMonitor", "build_monitor"]
