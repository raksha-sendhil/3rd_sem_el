"""Agent 3 metrics: leak rate, utility and routing accuracy.  OWNER: Person D.

All three are computed per document (paired by ``doc_id``) and micro-averaged.

* leak_rate: of the gold routes whose gold action is placeholder or generalise (QUASI_COMBO
  routes are skipped: their text is a description, not a string in the document), the share
  whose original text is still present in ``safe_text``.  "Present" uses the rewriter's own
  leak-check matcher (``leakcheck.contains``: case, spacing and format variants), so the
  metric and the retry loop agree on what a leak is.  A document with no prediction counts as
  no leak (nothing was published).
* utility: of the gold facts with importance "high", the share still recoverable from
  ``safe_text``.  A fact is recoverable when at least 60% of the content words of its value
  appear in the safe text (so a rephrasing such as "ringing in the left ear" still counts), or
  when a removed detail inside the value was replaced and that replacement appears in the safe
  text (the generalised form: "amyloidosis" -> "a rare protein disorder").  Documents with no
  prediction count as nothing recovered.
* routing_accuracy: of the gold routes, the share where the predicted decision for the same
  text (case-insensitive; repeated texts are matched in order) has the same action.  A gold
  route with no predicted decision counts as wrong.
* support: number of gold routes.

Empty inputs give 0.0 rather than raising.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque

from sanitise.agents.rewriter.leakcheck import contains
from sanitise.schemas import (
    Action,
    ExtractorOutput,
    GoldRouting,
    Importance,
    PIICategory,
    RewriteMetrics,
    RewriterOutput,
)

_WORD = re.compile(r"[a-z0-9]+")
_FILLER = frozenset(
    "a an the of in on at my his her their our its and with to for is was are".split()
)
THRESHOLD = 0.6


def _content(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _FILLER}


def _recoverable(fact_value: str, out: RewriterOutput) -> bool:
    safe = out.safe_text.lower()
    words = _content(fact_value)
    if words and len(words & _content(safe)) / len(words) >= THRESHOLD:
        return True
    value = fact_value.lower()
    for d in out.decisions:
        if d.action is Action.KEEP or not d.replacement:
            continue
        span = d.span_text.lower()
        if span and (span in value or value in span) and d.replacement.lower() in safe:
            return True
    return False


def score_rewrite(
    pred: list[RewriterOutput], gold_routing: list[GoldRouting], gold_facts: list[ExtractorOutput]
) -> RewriteMetrics:
    by_id = {p.doc_id: p for p in pred}
    facts_by_id = {g.doc_id: g.facts for g in gold_facts}
    removable = leaked = high = kept = routes = correct = 0
    for g in gold_routing:
        p = by_id.get(g.doc_id)
        queues: dict[str, deque] = defaultdict(deque)
        for d in p.decisions if p else []:
            queues[d.span_text.lower()].append(d.action)
        for route in g.routing:
            routes += 1
            q = queues.get(route.text.lower())
            correct += bool(q) and q.popleft() is route.action
            if route.action is Action.KEEP or route.category is PIICategory.QUASI_COMBO:
                continue
            removable += 1
            leaked += bool(p) and contains(p.safe_text, route.text, route.category)
        for fact in facts_by_id.get(g.doc_id, []):
            if fact.importance is Importance.HIGH:
                high += 1
                kept += bool(p) and _recoverable(fact.value, p)
    return RewriteMetrics(
        leak_rate=leaked / removable if removable else 0.0,
        utility=kept / high if high else 0.0,
        routing_accuracy=correct / routes if routes else 0.0,
        support=routes,
    )
