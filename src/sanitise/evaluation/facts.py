"""Agent 2 metrics: fact coverage and faithfulness.  OWNER: Person D (Data and Evaluation).

PLACEHOLDER created by Person C (a working first version). Refine the body; keep the
signature (docs/phase3_plan.md 6.2).

Matching rule used here: a predicted fact matches a gold fact of the same document when the
subject and relation are equal and the values share at least half of their words (token
Jaccard >= 0.5, case-insensitive). Matching is one-to-one, as in the Phase 2 span scorer.

* coverage      = matched gold facts / gold facts
* faithfulness  = matched predicted facts / predicted facts
* high_coverage = coverage over gold facts with importance "high" only
"""

from __future__ import annotations

import re

from sanitise.schemas import ExtractorOutput, FactMetrics, FactTuple, Importance

_WORD = re.compile(r"[a-z0-9]+")


def _jaccard(a: str, b: str) -> float:
    x, y = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    return len(x & y) / len(x | y) if x | y else 0.0


def _match(pred: list[FactTuple], gold: list[FactTuple]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    used_p: set[int] = set()
    for j, g in enumerate(gold):
        best, best_i = 0.0, None
        for i, p in enumerate(pred):
            if i in used_p or p.subject != g.subject or p.relation != g.relation:
                continue
            score = _jaccard(p.value, g.value)
            if score >= 0.5 and score > best:
                best, best_i = score, i
        if best_i is not None:
            used_p.add(best_i)
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
