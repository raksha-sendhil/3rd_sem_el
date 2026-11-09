"""Session privacy monitor (docs/session_monitor.md).  OWNER: Person C.

The protected pipeline calls ``monitor.check(session_id, query, hits, draft_answer)`` before
returning every answer (docs/phase3_plan.md 6.7):

* ``LedgerMonitor`` (ledger_monitor.py): the real monitor. It tracks, per session, which kinds
  of detail have been released about each person, and generalises or blocks an answer that
  would let the session single someone out (more than N attribute types, or fewer than k
  records still matching);
* ``AllowAllMonitor``: always allows, for comparison (``SANITISE_MONITOR=off``).

``build_monitor(settings, store)`` picks one from the settings.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sanitise.config import Settings
from sanitise.monitor.ledger import DisclosureLedger
from sanitise.monitor.ledger_monitor import LedgerMonitor
from sanitise.rag.store import DocumentStore
from sanitise.schemas import MonitorVerdict, RetrievalHit


@runtime_checkable
class SessionMonitor(Protocol):
    def check(
        self, session_id: str, query: str, hits: list[RetrievalHit], draft_answer: str
    ) -> MonitorVerdict: ...


class AllowAllMonitor:
    """No monitor: every answer is allowed (the Phase 3a stub, kept for comparison)."""

    def check(
        self, session_id: str, query: str, hits: list[RetrievalHit], draft_answer: str
    ) -> MonitorVerdict:
        return MonitorVerdict()


def build_monitor(settings: Settings, store: DocumentStore) -> SessionMonitor:
    if settings.monitor == "off":
        return AllowAllMonitor()
    # mock mode keeps its ledger in memory, so a demo leaves no file behind
    path = ":memory:" if settings.llm_backend == "mock" else settings.monitor_db_path
    return LedgerMonitor(
        store,
        k=settings.monitor_k,
        max_attributes=settings.monitor_max_attributes,
        max_climb=settings.monitor_max_climb,
        ledger=DisclosureLedger(path, settings.monitor_salt),
    )


__all__ = [
    "AllowAllMonitor",
    "DisclosureLedger",
    "LedgerMonitor",
    "SessionMonitor",
    "build_monitor",
]
