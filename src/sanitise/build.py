"""Batch processing: sanitise a corpus into the document store.  OWNER: Person C.

``build_store`` runs the offline path (Agents 1 -> 2 -> 3) over every document, writes each
original and its ``SanitisedRecord`` to the store, and optionally a JSON copy per document
(``data/processed/<doc_id>.json``). It resumes: documents already in the store are skipped, so
a Colab session that disconnects halfway loses at most the document in progress. A failure on
one document is recorded and the run continues.

    sanitise build --input data/corpus/docs --store data/store.sqlite --out data/processed
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from sanitise.agents.pipeline import sanitise
from sanitise.errors import PipelineError
from sanitise.llm.base import LLMClient
from sanitise.loader import Document
from sanitise.rag.store import DocumentStore
from sanitise.schemas import SanitisedRecord

log = logging.getLogger(__name__)

Sanitiser = Callable[[str, str, LLMClient], Awaitable[SanitisedRecord]]
Progress = Callable[[int, Document, str], None]  # (index, document, "ok" / "skipped" / error)


@dataclass
class BuildReport:
    processed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # doc_id -> error
    leaky: list[str] = field(default_factory=list)  # processed but residual_leaks non-empty
    seconds: float = 0.0

    @property
    def total(self) -> int:
        return len(self.processed) + len(self.skipped) + len(self.failed)

    def summary(self) -> str:
        per_doc = self.seconds / len(self.processed) if self.processed else 0.0
        return (
            f"{len(self.processed)} sanitised, {len(self.skipped)} already in the store, "
            f"{len(self.failed)} failed, {len(self.leaky)} with residual leaks "
            f"({self.seconds:.0f} s, {per_doc:.1f} s per document)"
        )


async def build_store(
    documents: Iterable[Document],
    store: DocumentStore,
    llm: LLMClient,
    *,
    out_dir: str | Path | None = None,
    resume: bool = True,
    sanitiser: Sanitiser = sanitise,
    progress: Progress | None = None,
) -> BuildReport:
    report = BuildReport()
    done = set(store.ids()) if resume else set()
    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for i, doc in enumerate(documents, start=1):
        if doc.doc_id in done:
            report.skipped.append(doc.doc_id)
            if progress:
                progress(i, doc, "skipped")
            continue
        try:
            record = await sanitiser(doc.doc_id, doc.text, llm)
        except PipelineError as exc:
            report.failed[doc.doc_id] = str(exc)
            log.warning("%s (%s): %s", doc.doc_id, doc.source, exc)
            if progress:
                progress(i, doc, str(exc))
            continue
        store.put(doc.doc_id, doc.text, record)
        if out is not None:
            (out / f"{doc.doc_id}.json").write_text(
                record.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
        report.processed.append(doc.doc_id)
        if record.residual_leaks:
            report.leaky.append(doc.doc_id)
        if progress:
            progress(i, doc, "ok")
    report.seconds = time.perf_counter() - t0
    return report
