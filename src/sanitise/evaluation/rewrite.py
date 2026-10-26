"""Agent 3 metrics: leak rate, utility and routing accuracy.  OWNER: Person D.

PLACEHOLDER created by Person C (a working first version). Refine the body; keep the
signature (docs/phase3_plan.md 6.2).

For each document (paired by ``doc_id``):

* leak_rate: of the gold spans whose gold action is placeholder or generalise, the share whose
  original text still appears in ``safe_text`` (case-insensitive substring).
* utility: of the gold facts with importance "high", the share whose value still appears in
  ``safe_text`` (case-insensitive substring). The real version should also count a fact as
  kept when its generalised form appears.
* routing_accuracy: of the gold routes, the share where the predicted decision for the same
  text (case-insensitive) has the same action. A gold route with no predicted decision counts
  as wrong.
"""

from __future__ import annotations

from sanitise.schemas import (
    Action,
    ExtractorOutput,
    GoldRouting,
    Importance,
    RewriteMetrics,
    RewriterOutput,
)


def score_rewrite(
    pred: list[RewriterOutput], gold_routing: list[GoldRouting], gold_facts: list[ExtractorOutput]
) -> RewriteMetrics:
    by_id = {p.doc_id: p for p in pred}
    facts_by_id = {g.doc_id: g.facts for g in gold_facts}
    removable = leaked = high = kept = routes = correct = 0
    for g in gold_routing:
        p = by_id.get(g.doc_id)
        safe = p.safe_text.lower() if p else ""
        actions = {d.span_text.lower(): d.action for d in p.decisions} if p else {}
        for route in g.routing:
            routes += 1
            correct += actions.get(route.text.lower()) is route.action
            if route.action is not Action.KEEP:
                removable += 1
                leaked += route.text.lower() in safe
        for fact in facts_by_id.get(g.doc_id, []):
            if fact.importance is Importance.HIGH:
                high += 1
                kept += fact.value.lower() in safe
    return RewriteMetrics(
        leak_rate=leaked / removable if removable else 0.0,
        utility=kept / high if high else 0.0,
        routing_accuracy=correct / routes if routes else 0.0,
        support=routes,
    )
