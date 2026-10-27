"""Retrieval over the ORIGINAL documents, returning document IDs only.  OWNER: Person C.

The index is built on the original text so retrieval quality is the same for both pipelines;
the protected pipeline then fetches the sanitised version of each hit (spec 1.1, "searches the
originals but serves only the rewritten text").

``BM25Retriever`` is the sparse half, in plain Python. The hybrid retriever (bge-m3 dense
vectors in ChromaDB, fused with BM25 by reciprocal rank fusion) implements the same
``Retriever`` protocol and replaces it without any caller changing.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol, runtime_checkable

from sanitise.schemas import RetrievalHit

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have he her his i in is it my of on or she that "
    "the their them they this to was were what which who with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, k: int = 3) -> list[RetrievalHit]: ...


class BM25Retriever:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._ids = list(docs)
        self._tf = [Counter(tokenize(docs[i])) for i in self._ids]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self._ids)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def __len__(self) -> int:
        return len(self._ids)

    def search(self, query: str, k: int = 3) -> list[RetrievalHit]:
        terms = tokenize(query)
        scored = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            for t in terms:
                if t in tf:
                    norm = self.k1 * (1 - self.b + self.b * self._len[i] / (self._avg or 1))
                    score += self._idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm)
            if score > 0:
                scored.append((score, self._ids[i]))
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [
            RetrievalHit(doc_id=doc_id, score=round(score, 4), sparse_rank=rank)
            for rank, (score, doc_id) in enumerate(scored[:k], start=1)
        ]
