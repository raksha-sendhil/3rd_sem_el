"""Helpers every finance fetcher uses to write its folder in the agreed format.

The format is the contract in docs/finance_plan.md ("Contracts"); ``tools/eval_suite/finance.py``
reads it. Use these functions instead of writing the files by hand, so every source matches:

    from common import Folder
    f = Folder("bank")            # data/finance/bank/
    f.reset()                     # remove the sample (or an earlier fetch)
    f.doc(text, private=["Asha Rao"], meta={"row": 17})            # docs/<id>.json
    f.doc(text, heldout=True)                                      # heldout/<id>.json
    f.utility("What fee applies to ...?", reference="...", doc_text=text)
    f.targeted("What is the account number of ...?", values=["50100234567"])
    f.scan(file_bytes, "stmt_001.pdf", type="bank_statement", fields={...}, private=[...],
           questions=[{"question": "...", "answer": "..."}], text=None)
    f.save()                      # writes utility.json, targeted.json, gold.jsonl
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINANCE = ROOT / "data" / "finance"
MAX_CHARS = 1900  # under the agents' 2,000-character limit


def doc_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class Folder:
    def __init__(self, source: str) -> None:
        self.source = source
        self.path = FINANCE / source
        self._utility: list[dict] = []
        self._targeted: list[dict] = []
        self._gold: list[dict] = []

    def reset(self) -> None:
        """Remove everything this source wrote before (the sample data included)."""
        for name in ("docs", "heldout", "scans", "population"):
            shutil.rmtree(self.path / name, ignore_errors=True)
        for name in ("utility.json", "targeted.json", "gold.jsonl"):
            (self.path / name).unlink(missing_ok=True)

    def doc(self, text: str, *, private: list[str] | None = None, meta: dict | None = None,
            heldout: bool = False, folder: str | None = None, person_id: str | None = None) -> str | None:  # fmt: skip
        """One text document. Returns its id, or None when it is too long or empty."""
        text = " ".join(text.split()) if "\n" not in text else text.strip()
        if not text or len(text) > MAX_CHARS:
            return None
        did = doc_id(text)
        row = {"doc_id": did, "text": text, "source": self.source, "private": private or [],
               "meta": meta or {}}  # fmt: skip
        if person_id:
            row["person_id"] = person_id
        sub = folder or ("heldout" if heldout else "docs")
        out = self.path / sub / f"{did}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(row, indent=1, ensure_ascii=False), encoding="utf-8")
        return did

    def utility(self, query: str, *, reference: str, doc_text: str | None = None) -> None:
        """A question about a document's content (never naming a person, company, account or
        place); ``reference`` is the correct answer, from the dataset."""
        did = doc_id(
            " ".join(doc_text.split())
            if doc_text and "\n" not in doc_text
            else (doc_text or "").strip()
        )
        self._utility.append({"id": f"u{len(self._utility):03d}", "query": query,
                              "reference": reference, "doc_id": did if doc_text else None})  # fmt: skip

    def targeted(self, query: str, *, values: list[str]) -> None:
        """A single question after labelled private values (Target Info)."""
        self._targeted.append({"id": f"t{len(self._targeted):03d}", "query": query,
                               "values": [v for v in values if v]})  # fmt: skip

    def scan(self, data: bytes, filename: str, *, type: str, fields: dict, private: list[str],
             questions: list[dict] | None = None, text: str | None = None) -> None:  # fmt: skip
        """One scanned document for the end-to-end check, labelled by the dataset's own fields.
        ``private`` names the fields that identify someone; ``text`` is a full transcription if
        the dataset has one (then run A, gold text, is built too)."""
        out = self.path / "scans" / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        row = {"id": Path(filename).stem, "file": f"scans/{filename}", "type": type,
               "fields": {k: str(v) for k, v in fields.items()}, "private": private,
               "questions": questions or []}  # fmt: skip
        if text:
            row["text"] = text
        self._gold.append(row)

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        if self._utility:
            (self.path / "utility.json").write_text(
                json.dumps(self._utility, indent=1, ensure_ascii=False), encoding="utf-8"
            )
        if self._targeted:
            (self.path / "targeted.json").write_text(
                json.dumps(self._targeted, indent=1, ensure_ascii=False), encoding="utf-8"
            )
        if self._gold:
            (self.path / "gold.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self._gold),
                encoding="utf-8",
            )
        n_docs = (
            len(list((self.path / "docs").glob("*.json"))) if (self.path / "docs").exists() else 0
        )
        print(
            f"{self.source}: {n_docs} documents, {len(self._utility)} utility questions, "
            f"{len(self._targeted)} Target Info questions, {len(self._gold)} scans -> {self.path}"
        )


def write_people(people: list[dict]) -> None:
    """The bank population (``data/finance/people.json``): one row per customer, with the fields
    of ``sanitise_finance.schemas.Person`` (missing ones left out) and ``doc_ids``."""
    out = FINANCE / "people.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(people, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(people)} people -> {out}")
