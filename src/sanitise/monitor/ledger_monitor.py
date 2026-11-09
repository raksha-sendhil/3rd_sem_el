"""The session privacy monitor (Phase 2 spec 3.3; docs/session_monitor.md).  OWNER: Person C.

Deterministic, no model involved. For every protected answer, before it leaves the API:

1. **What does the draft reveal?** The attributes of the retrieved records (``attributes.py``)
   whose shown form appears in the draft answer, each tied to the person it is about.
2. **Disclosure threshold.** With what this session was already told (the ledger), would any
   person now have more than ``max_attributes`` (N = 3) distinct attribute types known?
   Then the answer is **blocked**: generalising cannot take a revealed type back.
3. **Re-identification check.** For every person in the answer, count the records in the store
   whose own attributes are consistent with everything disclosed about that person so far plus
   this answer (``candidate_pool``). Fewer than ``k`` (5) means the session could single the
   person out.
4. **Router.** Over the pool limit, the newly revealed forms climb the generalisation ladder,
   one step at a time and at most ``max_climb`` steps ("a town in the Nilgiris district" ->
   "a town in Tamil Nadu"), until the pool is back to ``k`` or more: the answer is returned
   **generalised**. If that is not enough, it is **blocked**.

Only what is actually released is written to the ledger (as keyed hashes). The verdict reports
the riskiest person: the smallest pool and the attribute types known about them.

The pool is counted exactly over every record in the store, not over a retrieval top-k as the
spec sketches: a retriever can miss a matching record, which would under-count the pool.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from sanitise.monitor.attributes import Attribute, mentioned, record_attributes, replace_form
from sanitise.monitor.ledger import DisclosureLedger
from sanitise.rag.store import DocumentStore
from sanitise.schemas import MonitorVerdict, RetrievalHit, SanitisedRecord

Known = dict[str, set[str]]  # attribute_type -> value hashes


@dataclass
class _Subject:
    """One person in this answer: what was known before, and their newly revealed attributes."""

    known: Known
    released: list[Attribute]


class LedgerMonitor:
    def __init__(
        self,
        store: DocumentStore,
        *,
        k: int = 5,
        max_attributes: int = 3,
        max_climb: int = 2,
        ledger: DisclosureLedger | None = None,
    ) -> None:
        self.k = k
        self.max_attributes = max_attributes
        self.max_climb = max_climb
        self.ledger = ledger or DisclosureLedger()
        self._by_doc: dict[str, list[Attribute]] = {}
        # subject_id -> attribute_type -> hashes of every form on that person's ladders
        self._population: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self._rounds: dict[str, int] = defaultdict(int)
        for doc_id in store.ids():
            self.add_record(store.sanitised(doc_id))

    # ------------------------------------------------------------------ the store's population

    def add_record(self, record: SanitisedRecord) -> None:
        attributes = record_attributes(record)
        self._by_doc[record.doc_id] = attributes
        for a in attributes:
            self._population[a.subject_id][a.type].update(
                self.ledger.hash(a.type, form) for form in a.ladder
            )

    def pool(self, known: Known) -> int:
        """Records whose person is consistent with everything in ``known``."""
        return sum(
            1
            for types in self._population.values()
            if all(hashes <= types.get(kind, set()) for kind, hashes in known.items())
        )

    # ------------------------------------------------------------------ the check

    def check(
        self, session_id: str, query: str, hits: list[RetrievalHit], draft_answer: str
    ) -> MonitorVerdict:
        self._rounds[session_id] += 1
        round_ = self._rounds[session_id]
        subjects = self._released(session_id, hits, draft_answer)
        if not subjects:
            return MonitorVerdict(
                verdict="allowed",
                reason="no personal attribute about anyone in this answer",
            )

        # 1. threshold: a type once revealed cannot be taken back by generalising
        for s in subjects.values():
            types = set(s.known) | {a.type for a in s.released}
            if len(types) > self.max_attributes:
                new = sorted({a.type for a in s.released} - set(s.known))
                return MonitorVerdict(
                    verdict="blocked",
                    candidate_pool=self.pool(self._merged(s, {})),
                    disclosed=sorted(types),
                    reason=(
                        f"would reveal {', '.join(new)}: {len(types)} kinds of detail about one "
                        f"person in this session, the limit is {self.max_attributes}"
                    ),
                )

        # 2. candidate pool, climbing the ladder while it is too small
        level: dict[Attribute, int] = {a: 0 for s in subjects.values() for a in s.released}
        before = {sid: self.pool(self._merged(s, level)) for sid, s in subjects.items()}
        while True:
            pools = {sid: self.pool(self._merged(s, level)) for sid, s in subjects.items()}
            small = [sid for sid, p in pools.items() if p < self.k]
            if not small:
                break
            climbed = False
            for sid in small:
                for a in subjects[sid].released:
                    # what the session was already told cannot be taken back, so only new
                    # details climb; one past the ladder is the generic phrase (says nothing)
                    if self._is_new(subjects[sid], a) and level[a] < min(
                        len(a.ladder), self.max_climb
                    ):
                        level[a] += 1
                        climbed = True
            if not climbed:
                sid = min(pools, key=pools.get)
                return MonitorVerdict(
                    verdict="blocked",
                    candidate_pool=pools[sid],
                    disclosed=self._types(subjects[sid], level),
                    reason=(
                        f"only {pools[sid]} record{'s' if pools[sid] != 1 else ''} would still "
                        f"match what this session knows about one person, even with the details "
                        f"made less specific (k = {self.k})"
                    ),
                )

        # 3. release: rewrite what climbed, record what goes out
        revised = draft_answer
        for a, lvl in level.items():
            if lvl:
                revised = replace_form(revised, a.shown, a.form(lvl))
        if any(lvl and mentioned(revised, a.shown) for a, lvl in level.items()):
            return MonitorVerdict(
                verdict="blocked",
                candidate_pool=min(pools.values()),
                reason="the answer could not be made less specific",
            )
        self.ledger.record(
            session_id,
            round_,
            [
                (a.subject_id, a.type, self.ledger.hash(a.type, a.form(lvl)))
                for a, lvl in level.items()
                if a.reveals(lvl)
            ],
        )
        sid = min(pools, key=pools.get)
        climbed_types = sorted({a.type for a, lvl in level.items() if lvl})
        if climbed_types:
            return MonitorVerdict(
                verdict="generalised",
                candidate_pool=pools[sid],
                disclosed=self._types(subjects[sid], level),
                reason=(
                    f"{', '.join(climbed_types)} made less specific: {before[sid]} -> "
                    f"{pools[sid]} records match what this session knows (k = {self.k})"
                ),
                revised_answer=revised,
            )
        types = self._types(subjects[sid], level)
        return MonitorVerdict(
            verdict="allowed",
            candidate_pool=pools[sid],
            disclosed=types,
            reason=(
                f"{len(types)} of {self.max_attributes} kinds of detail known about one person; "
                f"{pools[sid]} records still match (k = {self.k})"
            ),
        )

    # ------------------------------------------------------------------ helpers

    def _released(
        self, session_id: str, hits: list[RetrievalHit], draft: str
    ) -> dict[str, _Subject]:
        known = self.ledger.known(session_id)
        out: dict[str, _Subject] = {}
        for hit in hits:
            for a in self._by_doc.get(hit.doc_id, []):
                if not mentioned(draft, a.shown):
                    continue
                s = out.setdefault(a.subject_id, _Subject(known.get(a.subject_id, {}), []))
                if a not in s.released:
                    s.released.append(a)
        return out

    def _merged(self, s: _Subject, level: dict[Attribute, int]) -> Known:
        merged: Known = {kind: set(hashes) for kind, hashes in s.known.items()}
        for a in s.released:
            lvl = level.get(a, 0)
            if a.reveals(lvl):
                merged.setdefault(a.type, set()).add(self.ledger.hash(a.type, a.form(lvl)))
        return merged

    def _is_new(self, s: _Subject, a: Attribute) -> bool:
        return self.ledger.hash(a.type, a.shown) not in s.known.get(a.type, set())

    @staticmethod
    def _types(s: _Subject, level: dict[Attribute, int]) -> list[str]:
        shown = {a.type for a in s.released if a.reveals(level.get(a, 0))}
        return sorted(set(s.known) | shown)

    def forget(self, session_id: str) -> None:
        self.ledger.forget(session_id)
        self._rounds.pop(session_id, None)
