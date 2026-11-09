"""The disclosure ledger (Phase 2 spec 3.3): what each session has been told, about whom.

SQLite, one row per released value::

    disclosures(session_id, subject_id, attribute_type, round, value_hash)

Values are never stored: ``value_hash`` is an HMAC-SHA256 of the attribute type and the
released phrase under a secret salt, so a copy of the ledger tells nobody what was disclosed,
yet the monitor can still match a released phrase against every record's own phrases (hashed
the same way). Without a configured salt a random one is drawn per process; the ledger then
only means something for the sessions of that process, which is how long sessions live anyway.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from sanitise.monitor.attributes import norm

SCHEMA = """
CREATE TABLE IF NOT EXISTS disclosures (
    session_id     TEXT NOT NULL,
    subject_id     TEXT NOT NULL,
    attribute_type TEXT NOT NULL,
    round          INTEGER NOT NULL,
    value_hash     TEXT NOT NULL,
    PRIMARY KEY (session_id, subject_id, attribute_type, value_hash)
);
CREATE INDEX IF NOT EXISTS disclosures_session ON disclosures (session_id);
"""


@dataclass(frozen=True)
class Disclosure:
    subject_id: str
    attribute_type: str
    round: int
    value_hash: str


class DisclosureLedger:
    def __init__(self, path: str | Path = ":memory:", salt: str | bytes | None = None) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()
        if salt is None:
            salt = os.urandom(32)
        self._salt = salt.encode("utf-8") if isinstance(salt, str) else salt

    def hash(self, attribute_type: str, value: str) -> str:
        message = f"{attribute_type}|{norm(value)}".encode()
        return hmac.new(self._salt, message, hashlib.sha256).hexdigest()

    def record(self, session_id: str, round_: int, items: list[tuple[str, str, str]]) -> None:
        """``items``: (subject_id, attribute_type, value_hash); a repeat keeps its first round."""
        with self._lock, self._db:
            self._db.executemany(
                "INSERT OR IGNORE INTO disclosures VALUES (?, ?, ?, ?, ?)",
                [(session_id, s, t, round_, h) for s, t, h in items],
            )

    def known(self, session_id: str) -> dict[str, dict[str, set[str]]]:
        """subject_id -> attribute_type -> value hashes, everything this session was told."""
        out: dict[str, dict[str, set[str]]] = {}
        for d in self.entries(session_id):
            out.setdefault(d.subject_id, {}).setdefault(d.attribute_type, set()).add(d.value_hash)
        return out

    def entries(self, session_id: str) -> list[Disclosure]:
        with self._lock:
            rows = self._db.execute(
                "SELECT subject_id, attribute_type, round, value_hash FROM disclosures "
                "WHERE session_id = ? ORDER BY round, subject_id, attribute_type",
                (session_id,),
            ).fetchall()
        return [Disclosure(*row) for row in rows]

    def forget(self, session_id: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM disclosures WHERE session_id = ?", (session_id,))

    def close(self) -> None:
        self._db.close()
