"""Agent 2 metrics: fact coverage and faithfulness.  OWNER: Person D (Data and Evaluation).

Matching rule (one-to-one within a document, as in the Phase 2 span scorer). A predicted fact
matches a gold fact when

* the subject ids are equal and the relations are equal, and
* the values agree after normalisation: lower-cased, plural ``s`` stripped, filler words
  (articles, ``my``, ``his``...) dropped, and then either the token sets have Jaccard
  similarity >= 0.5, or one set is wholly contained in the other and has at least two tokens
  (``34 year old`` matches ``34 years old``; ``pain`` does not match ``left sided abdominal
  pain``).

Each gold fact takes the best still-unmatched prediction, so a prediction is never counted
twice.  Documents with no prediction count as zero matches.

* coverage      = matched gold facts / gold facts
* faithfulness  = matched predicted facts / predicted facts.  The signature has no source
  text, so "supported by the text" is approximated by "agrees with the gold facts"; a correct
  fact the annotator did not list counts against faithfulness, which makes it a lower bound.
* high_coverage = coverage over gold facts with importance "high" only
* support       = number of gold facts

Empty inputs give 0.0 rather than raising.
"""

from __future__ import annotations

import re

from sanitise.schemas import ExtractorOutput, FactMetrics, FactTuple, Importance

_WORD = re.compile(r"[a-z0-9]+")
_FILLER = frozenset(
    "a an the of in on at my his her their our its and with to for is was are has have had".split()
)


def _tokens(value: str) -> frozenset[str]:
    out = set()
    for w in _WORD.findall(value.lower()):
        if w in _FILLER:
            continue
        out.add(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w)
    return frozenset(out)


def _similar(a: str, b: str) -> float:
    x, y = _tokens(a), _tokens(b)
    if not x or not y:
        return 1.0 if a.strip().lower() == b.strip().lower() and a.strip() else 0.0
    jaccard = len(x & y) / len(x | y)
    if (x <= y or y <= x) and min(len(x), len(y)) >= 2:
        return max(jaccard, 0.5)
    return jaccard


def _match(pred: list[FactTuple], gold: list[FactTuple]) -> list[tuple[int, int]]:
    """One-to-one pairs ``(pred index, gold index)``."""
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for j, g in enumerate(gold):
        best, best_i = 0.0, None
        for i, p in enumerate(pred):
            if i in used or p.subject != g.subject or p.relation != g.relation:
                continue
            score = _similar(p.value, g.value)
            if score >= 0.5 and score > best:
                best, best_i = score, i
        if best_i is not None:
            used.add(best_i)
            pairs.append((best_i, j))
    return pairs


def score_facts(pred: list[ExtractorOutput], gold: list[ExtractorOutput]) -> FactMetrics:
    by_id = {p.doc_id: p.facts for p in pred}
    n_gold = n_pred = matched = n_high = matched_high = 0
    for g in gold:
        p_facts = by_id.get(g.doc_id, [])
        pairs = _match(p_facts, g.facts)
        n_gold += len(g.facts)
        n_pred += len(p_facts)
        matched += len(pairs)
        n_high += sum(f.importance is Importance.HIGH for f in g.facts)
        matched_high += sum(g.facts[j].importance is Importance.HIGH for _, j in pairs)
    return FactMetrics(
        coverage=matched / n_gold if n_gold else 0.0,
        faithfulness=matched / n_pred if n_pred else 0.0,
        high_coverage=matched_high / n_high if n_high else 0.0,
        support=n_gold,
    )
