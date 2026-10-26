"""Session privacy monitor hook (docs/phase3_plan.md 6.7).  OWNER: Person C.

The protected pipeline calls ``monitor.check(...)`` before returning every answer. Phase 3a
ships ``AllowAllMonitor``, which always allows; the Phase 3b monitor replaces it and fills
``candidate_pool`` and ``disclosed``. Nothing else changes when it does: the API response
and the dashboard already carry those fields.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sanitise.schemas import MonitorVerdict, RetrievalHit


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


__all__ = ["AllowAllMonitor", "SessionMonitor"]
