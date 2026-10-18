"""Stage 1 - document load (spec section 3.1).

Reads a ChatDoctor or Enron record, assigns ``doc_id`` = the first 16 hex
characters of the SHA-256 of the raw text, and rejects documents longer than
``max_doc_chars`` (2000) with ``DOC_TOO_LONG``.

The text is used exactly as stored - no stripping, no newline translation - so
that character offsets computed by the detector, by the annotators and by the
scorer all refer to the same string.

Accepted inputs:

=====================  ==============================================================
``*.txt``              the whole file is the document
``*.json``             one record, or a list of records (pick one with ``index``)
``*.jsonl``            one record per line
``*.csv``              one record per row (the Kaggle Enron dump has ``file,message``)
a directory            every supported file inside it, sorted by name
=====================  ==============================================================

Record shapes (first match wins):

* fixture / gold:  ``{"text": ...}``
* ChatDoctor:      ``{"instruction": ..., "input": <patient>, "output": <doctor>}``
* Enron:           ``{"message": <raw email>}``, or ``{"body": ...}`` / ``{"content": ...}``
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from sanitise.config import Settings, get_settings
from sanitise.errors import DocTooLong

log = logging.getLogger(__name__)

SourceFormat = Literal["text", "fixture", "chatdoctor", "enron"]
SUPPORTED_SUFFIXES = (".json", ".jsonl", ".txt", ".csv")


class Document(BaseModel):
    doc_id: str
    text: str
    sha256: str
    source: str | None = None
    source_format: SourceFormat = "text"


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_doc_id(text: str) -> str:
    return sha256_hex(text)[:16]


def make_document(
    text: str,
    *,
    source: str | None = None,
    source_format: SourceFormat = "text",
    max_chars: int | None = None,
) -> Document:
    """Wrap raw text as a ``Document``. Raises ``DocTooLong`` past ``max_chars``."""
    limit = get_settings().max_doc_chars if max_chars is None else max_chars
    if len(text) > limit:
        where = f" ({source})" if source else ""
        raise DocTooLong(
            f"document{where} has {len(text)} characters; the limit is {limit}", stage="load"
        )
    digest = sha256_hex(text)
    return Document(
        doc_id=digest[:16], text=text, sha256=digest, source=source, source_format=source_format
    )


def extract_text(
    record: dict[str, Any], *, include_answer: bool = False
) -> tuple[str, SourceFormat]:
    """Pull the document text out of one dataset record."""
    if isinstance(record.get("text"), str):
        return record["text"], "fixture"
    if isinstance(record.get("input"), str) and ("output" in record or "instruction" in record):
        text = record["input"]
        if include_answer and isinstance(record.get("output"), str) and record["output"]:
            text = f"{text}\n\n{record['output']}"
        return text, "chatdoctor"
    for key in ("message", "body", "content"):
        if isinstance(record.get(key), str):
            return record[key], "enron"
    raise ValueError(
        "unrecognised record: expected a 'text' field (fixture), 'input'/'output' (ChatDoctor) "
        f"or 'message'/'body'/'content' (Enron); got keys {sorted(record)}"
    )


def _records(path: Path) -> list[dict[str, Any]] | None:
    """Records in a structured file, or ``None`` for plain text."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return None
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, list) else [data]
    if suffix == ".jsonl":
        with path.open(encoding="utf-8-sig") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    if suffix == ".csv":
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
        with path.open(encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))
    raise ValueError(f"{path}: unsupported file type (supported: {', '.join(SUPPORTED_SUFFIXES)})")


def _document_from_record(record: dict[str, Any], source: str, settings: Settings) -> Document:
    if not isinstance(record, dict):
        raise ValueError(f"{source}: record is a {type(record).__name__}, expected an object")
    text, fmt = extract_text(record, include_answer=settings.chatdoctor_include_answer)
    doc = make_document(text, source=source, source_format=fmt, max_chars=settings.max_doc_chars)
    stated = record.get("doc_id")
    if stated and stated != doc.doc_id:
        log.warning(
            "%s: stored doc_id %s differs from sha256-derived %s; using %s",
            source,
            stated,
            doc.doc_id,
            doc.doc_id,
        )
    return doc


def load_document(
    path: str | Path, *, index: int = 0, settings: Settings | None = None
) -> Document:
    """Load one document from a file (record ``index`` for multi-record files)."""
    settings = settings or get_settings()
    path = Path(path)
    records = _records(path)
    if records is None:
        with path.open(encoding="utf-8-sig", newline="") as fh:
            text = fh.read()
        return make_document(text, source=str(path), max_chars=settings.max_doc_chars)
    if not records:
        raise ValueError(f"{path}: file contains no records")
    if not -len(records) <= index < len(records):
        raise IndexError(f"{path}: record index {index} out of range (0..{len(records) - 1})")
    source = str(path) if len(records) == 1 else f"{path}#{index}"
    return _document_from_record(records[index], source, settings)


def iter_documents(
    path: str | Path, *, settings: Settings | None = None, skip_too_long: bool = False
) -> Iterator[Document]:
    """Every document under ``path`` (a file or a directory), in a stable order."""
    settings = settings or get_settings()
    path = Path(path)
    files = (
        sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
        if path.is_dir()
        else [path]
    )
    for file in files:
        records = _records(file)
        try:
            if records is None:
                yield load_document(file, settings=settings)
                continue
            for i, record in enumerate(records):
                source = str(file) if len(records) == 1 else f"{file}#{i}"
                try:
                    yield _document_from_record(record, source, settings)
                except DocTooLong as exc:
                    if not skip_too_long:
                        raise
                    log.warning("skipping: %s", exc.detail)
        except DocTooLong as exc:
            if not skip_too_long:
                raise
            log.warning("skipping: %s", exc.detail)
