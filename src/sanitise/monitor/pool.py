"""Session privacy monitor (Phase 3b).  OWNER: Person A (Deterministic Logic).

Legal build spec, step 12: count the clients that still match everything revealed in the
session so far, and block the answer when fewer than ``min_pool`` (5) remain. Plain Python,
no LLM.

What counts as revealed: every attribute value from the client table that appears in the
question or in the draft answer (the attacker knows their own question, and the answer
confirms or extends it). Values are matched against the table's own vocabulary, so a
number such as "1881" (the year of the NI Act) constrains nothing unless some client filed
in 1881. Within one attribute the values a session mentions are alternatives (a client
matches if it has any of them); across attributes they combine.

``check`` never commits a blocked round, so the pool reported to the session never drops
below ``min_pool``: the attacker learns nothing from a refusal except that it was refused.

The client table is ``data/clients.json`` (legal build spec, "Data"): a list of objects with
``city``, ``district``, ``state``, ``occupation``, ``occupation_group``, ``legal_issue``,
``court``, ``filing_date`` and ``age``. The medical person table's ``town`` and
``conditions`` fields are read too, so the sample data works unchanged.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from sanitise.agents.rewriter.ontology import court_type
from sanitise.schemas import MonitorVerdict, RetrievalHit

log = logging.getLogger(__name__)

DEFAULT_MIN_POOL = 5
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def _norm(text: str) -> str:
    return " " + _NON_ALNUM.sub(" ", text.lower()).strip() + " "


def _values(*fields: str) -> Callable[[dict[str, Any]], set[str]]:
    def read(client: dict[str, Any]) -> set[str]:
        out: set[str] = set()
        for f in fields:
            value = client.get(f)
            for v in value if isinstance(value, list) else [value]:
                if isinstance(v, str) and v.strip():
                    out.add(_norm(v).strip())
        return out

    return read


def _court(client: dict[str, Any]) -> set[str]:
    court = client.get("court")
    if not isinstance(court, str) or not court.strip():
        return set()
    return {re.sub(r"^(?:a|an|the) ", "", _norm(court_type(court)).strip())}


def _filing_year(client: dict[str, Any]) -> set[str]:
    match = re.search(r"(?:19|20)\d{2}", str(client.get("filing_date") or ""))
    return {match.group()} if match else set()


def _age_band(client: dict[str, Any]) -> set[str]:
    age = client.get("age")
    return {f"{int(age) // 10 * 10}s"} if isinstance(age, int | float) else set()


# attribute -> how to read a client's values. Order is the order ``disclosed`` lists them in.
ATTRIBUTES: dict[str, Callable[[dict[str, Any]], set[str]]] = {
    "legal_issue": _values("legal_issue", "conditions"),
    "state": _values("state"),
    "district": _values("district"),
    "city": _values("city", "town"),
    "occupation": _values("occupation"),
    "occupation_group": _values("occupation_group"),
    "court_type": _court,
    "filing_year": _filing_year,
    "age_band": _age_band,
}

_AGE_PHRASES = [
    re.compile(r"\b([1-9])0\s*['’]?s\b"),  # "in their 50s"
    re.compile(r"\b(\d{2})[\s-]*(?:years?|yrs?)[\s-]*old\b", re.I),
    re.compile(r"\baged?\s+(\d{2})\b", re.I),
]


def _ages(text: str) -> set[str]:
    found = set()
    for pattern in _AGE_PHRASES:
        for m in pattern.finditer(text):
            number = m.group(1)
            found.add(f"{number}0s" if len(number) == 1 else f"{int(number) // 10 * 10}s")
    return found


Disclosure = dict[str, set[str]]


class CandidatePoolMonitor:
    """Blocks a protected answer when it would leave fewer than ``min_pool`` clients."""

    def __init__(self, clients: Iterable[dict[str, Any]], min_pool: int = DEFAULT_MIN_POOL):
        self.min_pool = min_pool
        # each client's values per attribute, computed once
        self._clients = [{a: read(c) for a, read in ATTRIBUTES.items()} for c in clients]
        self._vocab: dict[str, set[str]] = {a: set() for a in ATTRIBUTES}
        for client in self._clients:
            for attribute, values in client.items():
                self._vocab[attribute] |= values
        self._sessions: dict[str, Disclosure] = {}

    @classmethod
    def from_file(cls, path: str | Path, min_pool: int = DEFAULT_MIN_POOL) -> CandidatePoolMonitor:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), min_pool)

    # ------------------------------------------------------------------ disclosure

    def disclosed_in(self, text: str) -> Disclosure:
        """Attribute values from the client table that ``text`` mentions."""
        haystack = _norm(text)
        found: Disclosure = {}
        for attribute, vocab in self._vocab.items():
            if attribute == "age_band":
                hits = _ages(text) & vocab
            else:
                # "land disputes" and "doctors" disclose "land dispute" and "doctor"
                hits = {v for v in vocab if any(f" {v}{s} " in haystack for s in ("", "s", "es"))}
            if hits:
                found[attribute] = hits
        return found

    def pool(self, disclosure: Disclosure) -> int:
        return sum(
            all(client[a] & values for a, values in disclosure.items()) for client in self._clients
        )

    def session(self, session_id: str) -> Disclosure:
        return {a: set(v) for a, v in self._sessions.get(session_id, {}).items()}

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    # ------------------------------------------------------------------ the hook

    def check(
        self, session_id: str, query: str, hits: list[RetrievalHit], draft_answer: str
    ) -> MonitorVerdict:
        committed = self.session(session_id)
        merged = self.session(session_id)
        for attribute, values in self.disclosed_in(f"{query}\n{draft_answer}").items():
            merged.setdefault(attribute, set()).update(values)

        after = self.pool(merged)
        names = [a for a in ATTRIBUTES if a in merged]
        if merged != committed and after < self.min_pool:
            before = self.pool(committed)
            return MonitorVerdict(
                verdict="blocked",
                candidate_pool=before,
                disclosed=[a for a in ATTRIBUTES if a in committed],
                reason=(
                    f"answering would leave {after} matching client(s) "
                    f"(minimum {self.min_pool}); disclosing {', '.join(names)}"
                ),
            )
        self._sessions[session_id] = merged
        return MonitorVerdict(
            verdict="allowed",
            candidate_pool=after,
            disclosed=names,
            reason=f"{after} clients still match everything disclosed in this session",
        )


def build_monitor(clients_path: str | Path, min_pool: int = DEFAULT_MIN_POOL):
    """The candidate-pool monitor over ``clients_path``, or the allow-all stub if it is missing."""
    from sanitise.monitor import AllowAllMonitor

    path = Path(clients_path)
    if not path.is_file():
        log.warning("client table %s not found: the session monitor allows every answer", path)
        return AllowAllMonitor()
    return CandidatePoolMonitor.from_file(path, min_pool)
