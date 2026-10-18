"""Stage 6 - merge.  OWNER: Person A (Deterministic Logic).

PLACEHOLDER created by Person C: a plain union so the pipeline runs. Replace
with the real merge - drop quasi-identifiers that merely restate an explicit
identifier, resolve overlaps by the fixed category precedence order, and merge
adjacent spans of the same category. Keep the signature.
"""

from __future__ import annotations

from sanitise.schemas import SensitiveSpan


def merge_spans(
    rule_spans: list[SensitiveSpan], llm_spans: list[SensitiveSpan]
) -> list[SensitiveSpan]:
    """Union, drop quasi-identifiers that restate an explicit identifier,
    resolve overlaps by category precedence."""
    # TODO(Person A): restatement filter, category precedence, adjacent-span merge.
    return [*rule_spans, *llm_spans]
