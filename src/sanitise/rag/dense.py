"""Dense retrieval and hybrid fusion.  OWNER: Person C.

* ``SentenceTransformerEmbedder`` - BAAI/bge-m3 through sentence-transformers (normalised
  vectors, so a dot product is the cosine similarity).
* ``ChromaDenseIndex`` - vectors in a persistent ChromaDB collection under ``index_dir``;
  ``MemoryDenseIndex`` - the same in memory (tests, small corpora).
* ``HybridRetriever`` - BM25 and dense rankings fused by reciprocal rank fusion
  (score = sum over rankings of 1 / (rrf_k + rank)). Both rankings are over the ORIGINAL
  documents and the result is document IDs only.
* ``build_retriever`` - hybrid when sentence-transformers and chromadb are installed (Colab),
  BM25 alone otherwise (a laptop without the 2 GB embedding model), with a log line saying so.

sentence-transformers and chromadb are optional dependencies (``pip install -e .[retrieval]``),
imported only when a dense index is actually built.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from sanitise.rag.retrieve import BM25Retriever, Retriever
from sanitise.rag.store import DocumentStore
from sanitise.schemas import RetrievalHit

log = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True, batch_size=16)
        return [list(map(float, v)) for v in vectors]


class DenseIndex(Protocol):
    def add(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None: ...
    def ids(self) -> set[str]: ...
    def search(self, query: str, k: int) -> list[tuple[str, float]]: ...


class MemoryDenseIndex:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._ids: list[str] = []
        self._vecs: list[list[float]] = []

    def add(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        if doc_ids:
            self._ids.extend(doc_ids)
            self._vecs.extend(self.embedder.embed(texts))

    def ids(self) -> set[str]:
        return set(self._ids)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        q = self.embedder.embed([query])[0]
        scored = [
            (sum(a * b for a, b in zip(q, v, strict=True)), d)
            for d, v in zip(self._ids, self._vecs, strict=True)
        ]
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [(d, s) for s, d in scored[:k]]


class ChromaDenseIndex:
    """A persistent ChromaDB collection; survives restarts, so indexing happens once."""

    def __init__(self, persist_dir: str | Path, embedder: Embedder, name: str = "originals"):
        import chromadb

        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._col = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        for start in range(0, len(doc_ids), 64):
            ids = list(doc_ids[start : start + 64])
            batch = list(texts[start : start + 64])
            self._col.upsert(ids=ids, embeddings=self.embedder.embed(batch))

    def ids(self) -> set[str]:
        return set(self._col.get(include=[])["ids"])

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._col.count() == 0:
            return []
        res = self._col.query(
            query_embeddings=self.embedder.embed([query]), n_results=min(k, self._col.count())
        )
        return [(d, 1.0 - dist) for d, dist in zip(res["ids"][0], res["distances"][0], strict=True)]


class HybridRetriever:
    def __init__(
        self, sparse: BM25Retriever, dense: DenseIndex, rrf_k: int = 60, depth: int = 20
    ) -> None:
        self.sparse, self.dense, self.rrf_k, self.depth = sparse, dense, rrf_k, depth

    def search(self, query: str, k: int = 4) -> list[RetrievalHit]:
        sparse = [h.doc_id for h in self.sparse.search(query, self.depth)]
        dense = [d for d, _ in self.dense.search(query, self.depth)]
        sparse_rank = {d: r for r, d in enumerate(sparse, start=1)}
        dense_rank = {d: r for r, d in enumerate(dense, start=1)}
        fused = {
            d: sum(1.0 / (self.rrf_k + rank[d]) for rank in (sparse_rank, dense_rank) if d in rank)
            for d in set(sparse) | set(dense)
        }
        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [
            RetrievalHit(
                doc_id=d,
                score=round(score, 6),
                dense_rank=dense_rank.get(d),
                sparse_rank=sparse_rank.get(d),
            )
            for d, score in ordered
        ]


def dense_available() -> bool:
    return all(importlib.util.find_spec(m) for m in ("sentence_transformers", "chromadb"))


def sync_index(index: DenseIndex, store: DocumentStore) -> int:
    """Embed every stored original the index does not have yet. Returns how many were added."""
    missing = [d for d in store.ids() if d not in index.ids()]
    index.add(missing, [store.original(d) for d in missing])
    return len(missing)


def build_retriever(
    store: DocumentStore,
    *,
    mode: str = "hybrid",
    index_dir: str | Path = "data/index",
    model_name: str = "BAAI/bge-m3",
    embedder: Embedder | None = None,
) -> Retriever:
    sparse = BM25Retriever({d: store.original(d) for d in store.ids()})
    if mode != "hybrid":
        return sparse
    if embedder is None and not dense_available():
        log.warning("sentence-transformers / chromadb not installed: using BM25 retrieval only")
        return sparse
    embedder = embedder or SentenceTransformerEmbedder(model_name)
    dense = (
        ChromaDenseIndex(index_dir, embedder) if dense_available() else MemoryDenseIndex(embedder)
    )
    added = sync_index(dense, store)
    if added:
        log.info("embedded %d new documents into the dense index", added)
    return HybridRetriever(sparse, dense)
