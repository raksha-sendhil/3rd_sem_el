"""Dual-track storage: every original next to its SanitisedRecord.  OWNER: Person C.

``SqliteDocumentStore`` is the persistent store the batch runner fills and the API reads;
``MemoryDocumentStore`` backs mock mode and tests. Both implement ``DocumentStore``
(docs/phase3_plan.md 6.2).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Protocol, runtime_checkable

from sanitise.errors import DocNotFound
from sanitise.schemas import SanitisedRecord


@runtime_checkable
class DocumentStore(Protocol):
    def put(self, doc_id: str, original: str, record: SanitisedRecord) -> None: ...
    def original(self, doc_id: str) -> str: ...
    def sanitised(self, doc_id: str) -> SanitisedRecord: ...
    def ids(self) -> list[str]: ...


class MemoryDocumentStore:
    def __init__(self) -> None:
        self._docs: dict[str, tuple[str, SanitisedRecord]] = {}

    def put(self, doc_id: str, original: str, record: SanitisedRecord) -> None:
        self._docs[doc_id] = (original, record)

    def original(self, doc_id: str) -> str:
        return self._get(doc_id)[0]

    def sanitised(self, doc_id: str) -> SanitisedRecord:
        return self._get(doc_id)[1]

    def ids(self) -> list[str]:
        return sorted(self._docs)

    def _get(self, doc_id: str) -> tuple[str, SanitisedRecord]:
        try:
            return self._docs[doc_id]
        except KeyError:
            raise DocNotFound(f"no document {doc_id!r}", stage="store") from None


class SqliteDocumentStore:
    """One table: doc_id, original text, SanitisedRecord as JSON."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS documents ("
            "doc_id TEXT PRIMARY KEY, original TEXT NOT NULL, record TEXT NOT NULL)"
        )
        self._db.commit()

    def put(self, doc_id: str, original: str, record: SanitisedRecord) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO documents VALUES (?, ?, ?)",
            (doc_id, original, record.model_dump_json()),
        )
        self._db.commit()

    def original(self, doc_id: str) -> str:
        return self._row(doc_id)[0]

    def sanitised(self, doc_id: str) -> SanitisedRecord:
        return SanitisedRecord.model_validate_json(self._row(doc_id)[1])

    def ids(self) -> list[str]:
        return [r[0] for r in self._db.execute("SELECT doc_id FROM documents ORDER BY doc_id")]

    def close(self) -> None:
        self._db.close()

    def _row(self, doc_id: str) -> tuple[str, str]:
        row = self._db.execute(
            "SELECT original, record FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            raise DocNotFound(f"no document {doc_id!r}", stage="store")
        return row
