"""Batch build (resume, failures) and hybrid retrieval (RRF fusion) without real models."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import pytest

from sanitise.build import build_store
from sanitise.errors import SchemaValidationFailed
from sanitise.loader import make_document
from sanitise.mock_mode import mock_llm, mock_sanitise
from sanitise.rag.dense import HybridRetriever, MemoryDenseIndex, build_retriever, sync_index
from sanitise.rag.retrieve import BM25Retriever
from sanitise.rag.store import MemoryDocumentStore, SqliteDocumentStore
from tests.conftest import SAMPLE_TEXT


class BagOfWords:
    """A tiny deterministic embedder: hashed word counts, L2-normalised."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * 64
            for word in text.lower().split():
                vec[int(hashlib.md5(word.strip(".,?").encode()).hexdigest(), 16) % 64] += 1
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


DOCS = {
    "a": "The lighthouse keeper has a rash on both hands.",
    "b": "A teacher asks about migraine and poor sleep.",
    "c": "My son plays kabaddi and tore a knee ligament.",
}


def test_hybrid_fuses_both_rankings() -> None:
    dense = MemoryDenseIndex(BagOfWords())
    dense.add(list(DOCS), list(DOCS.values()))
    hybrid = HybridRetriever(BM25Retriever(DOCS), dense)
    hits = hybrid.search("kabaddi knee ligament", k=2)
    assert hits[0].doc_id == "c"
    assert hits[0].sparse_rank == 1 and hits[0].dense_rank == 1
    assert hits[0].score == pytest.approx(2 / 61, abs=1e-6)  # stored to 6 decimals


async def test_build_store_resumes_and_syncs_index(fake_rules: None, tmp_path) -> None:
    store = SqliteDocumentStore(tmp_path / "s.sqlite")
    docs = [make_document(SAMPLE_TEXT), make_document("A second, harmless document.")]
    first = await build_store(
        docs, store, mock_llm(), out_dir=tmp_path / "out", sanitiser=mock_sanitise
    )
    assert len(first.processed) == 2 and not first.failed
    assert len(list((tmp_path / "out").glob("*.json"))) == 2
    again = await build_store(docs, store, mock_llm(), sanitiser=mock_sanitise)
    assert again.processed == [] and len(again.skipped) == 2
    index = MemoryDenseIndex(BagOfWords())
    assert sync_index(index, store) == 2 and sync_index(index, store) == 0
    retriever = build_retriever(store, mode="hybrid", embedder=BagOfWords())
    assert retriever.search("lighthouse keeper", k=1)[0].doc_id == docs[0].doc_id
    store.close()


async def test_a_failing_document_does_not_stop_the_run(fake_rules: None) -> None:
    async def flaky(doc_id: str, text: str, llm):
        if "boom" in text:
            raise SchemaValidationFailed("bad output", stage="extract")
        return await mock_sanitise(doc_id, text, llm)

    docs = [make_document("boom"), make_document(SAMPLE_TEXT)]
    store = MemoryDocumentStore()
    report = await build_store(docs, store, mock_llm(), sanitiser=flaky)
    assert list(report.failed) == [docs[0].doc_id] and report.processed == [docs[1].doc_id]
    assert "1 failed" in report.summary()


def test_bm25_only_mode() -> None:
    store = MemoryDocumentStore()
    assert isinstance(build_retriever(store, mode="bm25"), BM25Retriever)
