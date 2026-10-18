"""Stages 2-3 - deterministic pass.  OWNER: Person A (Deterministic Logic).

PLACEHOLDER created by Person C so the pipeline, CLI and API run end to end on
Day 1. Replace the body of ``rule_detect`` with Presidio Analyzer (spaCy
``en_core_web_lg``) plus the custom regex recognizers. Keep the signature.

Contract (spec 8.2), checked by ``tests/contract/test_interfaces.py`` and again
at runtime by ``detect()``:

* every span has ``method="rule"`` and exact ``start``/``end`` offsets;
* ``text[start:end]`` equals ``span.text`` (case/whitespace-insensitive);
* the function is synchronous and safe to call from a worker thread -
  ``detect()`` runs it via ``asyncio.to_thread`` so the API stays responsive.
  Build the ``AnalyzerEngine`` once (lazily, module level) rather than per call.
"""

from __future__ import annotations

from sanitise.schemas import SensitiveSpan


def rule_detect(text: str) -> list[SensitiveSpan]:
    """Presidio + spaCy + regex. Every span has exact start/end and method='rule'."""
    # TODO(Person A): Presidio + spaCy + custom regex recognizers.
    return []
