"""Session privacy monitor (Phase 3b).  OWNER: Person A (Deterministic Logic).

Legal build spec, step 12: count who still matches everything revealed in the session, and stop
fewer than ``min_pool`` (5) remaining. Plain Python, no LLM.

The monitor only knows what the system itself stores: the sanitised records. It has no client
table, so it cannot know who an evaluation will attack.

* The population is the store. Each record's attributes are the forms its rewrite shows
  ("a city in Karnataka", "in their 50s", "a district court"), never a removed value.
* Revealed = those forms, of the retrieved records, in the draft answer. The question counts
  only when the draft confirms or denies it ("Yes", "No, ..."): otherwise naming a detail in a
  question would be taken for a disclosure.
* The legal issue and a medical condition are content: never edited, and never the reason for
  a block when nothing personal is known. They do narrow the pool.
* A new personal detail that leaves fewer than ``min_pool`` records is replaced by its generic
  phrase (**generalised**). When that is impossible or not enough, the answer is **blocked**.
  Only what is released is remembered.
"""

from __future__ import annotations

import re

from sanitise_legal.agents.rewriter.ontology import _GENERIC
from sanitise_legal.rag.store import DocumentStore
from sanitise_legal.schemas import (
    Action,
    MonitorVerdict,
    PIICategory,
    RetrievalHit,
    SanitisedRecord,
)

C = PIICategory
DEFAULT_MIN_POOL = 5
CONTENT_TYPES = frozenset({"legal_issue", "condition"})
TYPE_OF = {
    C.LOCATION: "location",
    C.OCCUPATION: "occupation",
    C.AGE: "age",
    C.DATE: "date",
    C.COURT: "court",
    C.ORGANIZATION: "organisation",
    C.MONEY: "amount",
    C.PHYSICAL: "physical",
    C.LEGAL_ISSUE: "legal_issue",
    C.MEDICAL_CONDITION: "condition",
}
TYPES = tuple(TYPE_OF.values())
_GENERIC_OF = {kind: _GENERIC.get(c, "an adult") for c, kind in TYPE_OF.items()}
_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_LEAD = re.compile(
    r"^(?:in (?:their|his|her) |(?:a|an|the) )?"
    r"(?:(?:town|city|village|place|district|state) (?:in|of|near) (?:the )?)?"
)
_CONFIRMS = re.compile(
    r"^\W*(?:yes|no|correct|that(?: is|'s) (?:right|correct|true)|confirmed)\b", re.IGNORECASE
)


def _norm(text: str) -> str:
    return _NON_ALNUM.sub(" ", text.lower()).strip()


_SAYS_NOTHING = {_norm(p) for p in _GENERIC.values()} | {"an adult", "a person", "india"}


def _variants(form: str) -> list[str]:
    """The form and the core a model tends to repeat: "Karnataka" for "a city in Karnataka"."""
    core = _LEAD.sub("", form)
    return [form, core] if core != form and len(core) >= 3 else [form]


def _mentioned(text: str, form: str) -> bool:
    said = f" {_norm(text)} "
    return any(f" {v} " in said for v in _variants(form))


def _replace(text: str, form: str, new: str) -> str:
    for variant in _variants(form):
        words = r"[^0-9A-Za-z]+".join(re.escape(w) for w in variant.split())
        text = re.sub(
            rf"(?<![0-9A-Za-z]){words}(?![0-9A-Za-z])", lambda _: new, text, flags=re.IGNORECASE
        )
    return text


Disclosure = dict[str, set[str]]


def record_attributes(record: SanitisedRecord) -> Disclosure:
    """What the record's sanitised text shows, by type."""
    out: Disclosure = {}
    for d in record.rewriter.decisions:
        kind = TYPE_OF.get(d.category)
        if kind is None or d.start is None or d.action is Action.PLACEHOLDER:
            continue
        if d.action is Action.KEEP and kind == "amount":
            continue  # an amount that is the content of a contract clause
        shown = _norm(d.span_text if d.action is Action.KEEP else d.replacement or "")
        if shown and shown not in _SAYS_NOTHING:
            out.setdefault(kind, set()).add(shown)
    return out


class CandidatePoolMonitor:
    """Stops a protected answer that would leave fewer than ``min_pool`` matching records."""

    def __init__(self, store: DocumentStore, min_pool: int = DEFAULT_MIN_POOL):
        self.min_pool = min_pool
        self.store = store
        self._records: dict[str, Disclosure] = {}
        self._sessions: dict[str, dict[str, Disclosure]] = {}
        self._refresh()

    def _refresh(self) -> None:
        for doc_id in self.store.ids():
            if doc_id not in self._records:
                self._records[doc_id] = record_attributes(self.store.sanitised(doc_id))

    def pool(self, disclosure: Disclosure) -> int:
        return sum(
            all(values <= record.get(kind, set()) for kind, values in disclosure.items())
            for record in self._records.values()
        )

    def session(self, session_id: str) -> dict[str, Disclosure]:
        return {
            doc: {kind: set(values) for kind, values in known.items()}
            for doc, known in self._sessions.get(session_id, {}).items()
        }

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    # ------------------------------------------------------------------ the hook

    def check(
        self, session_id: str, query: str, hits: list[RetrievalHit], draft_answer: str
    ) -> MonitorVerdict:
        if any(h.doc_id not in self._records for h in hits):
            self._refresh()  # the store grew after the monitor was built
        known = self.session(session_id)
        said = f"{query}\n{draft_answer}" if _CONFIRMS.match(draft_answer) else draft_answer
        released: dict[str, Disclosure] = {}
        swaps: dict[str, str] = {}
        for hit in hits:
            doc = hit.doc_id
            before = known.get(doc, {})
            new: Disclosure = {}
            for kind, forms in self._records.get(doc, {}).items():
                for form in forms - before.get(kind, set()):
                    if _mentioned(said, form):
                        new.setdefault(kind, set()).add(form)
            if not new:
                continue
            if self._unsafe(before, new):
                for kind in [k for k in new if k not in CONTENT_TYPES]:
                    for form in new.pop(kind):
                        if not _mentioned(draft_answer, form):
                            # only the question says it: confirming cannot be made vaguer
                            return self._blocked(before, "confirming a detail in the question")
                        swaps[form] = _GENERIC_OF[kind]
                if self._unsafe(before, new):
                    return self._blocked(before, "even with the new details made generic")
            released[doc] = new

        revised = draft_answer
        for form in sorted(swaps, key=len, reverse=True):
            revised = _replace(revised, form, swaps[form])
        if any(_mentioned(revised, form) for form in swaps):
            return self._blocked({}, "a detail that cannot be made less specific")
        if swaps:  # what another record released is gone from the revised answer too
            for new in released.values():
                for kind in list(new):
                    new[kind] -= set(swaps)
                    if not new[kind]:
                        del new[kind]

        merged = {doc: self._merge(known.get(doc, {}), new) for doc, new in released.items()}
        merged = {doc: d for doc, d in merged.items() if d}
        self._sessions[session_id] = {**known, **merged}
        pool, names = None, []
        if merged:
            pool, riskiest = min(
                ((self.pool(d), d) for d in merged.values()), key=lambda found: found[0]
            )
            names = [t for t in TYPES if t in riskiest]
        if swaps:
            return MonitorVerdict(
                verdict="generalised",
                candidate_pool=pool,
                disclosed=names,
                reason=f"details made generic: fewer than {self.min_pool} records would match",
                revised_answer=revised,
            )
        return MonitorVerdict(
            verdict="allowed",
            candidate_pool=pool,
            disclosed=names,
            reason=(
                "no attribute of a retrieved record in this answer"
                if pool is None
                else f"{pool} records still match everything disclosed in this session"
            ),
        )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _merge(before: Disclosure, new: Disclosure) -> Disclosure:
        merged = {kind: set(values) for kind, values in before.items()}
        for kind, values in new.items():
            merged.setdefault(kind, set()).update(values)
        return merged

    def _unsafe(self, before: Disclosure, new: Disclosure) -> bool:
        if not new:
            return False  # nothing more is revealed
        merged = self._merge(before, new)
        personal = set(merged) - CONTENT_TYPES
        return bool(personal) and self.pool(merged) < self.min_pool

    def _blocked(self, before: Disclosure, why: str) -> MonitorVerdict:
        return MonitorVerdict(
            verdict="blocked",
            candidate_pool=self.pool(before),
            disclosed=[t for t in TYPES if t in before],
            reason=f"fewer than {self.min_pool} records would match ({why})",
        )


def build_monitor(store: DocumentStore, min_pool: int = DEFAULT_MIN_POOL) -> CandidatePoolMonitor:
    """The candidate-pool monitor over the store's sanitised records."""
    return CandidatePoolMonitor(store, min_pool)
