"""Stages 2-3 - deterministic pass.  OWNER: Person A (Deterministic Logic).

The implementation lives in the ``sanitise.rules`` package (Presidio + spaCy +
custom regex layers, normalisation, category precedence). This module is the
seam ``detect()`` imports from, so the pipeline does not depend on how that
package is laid out internally.

Contract (spec 8.2), checked by ``tests/contract/test_interfaces.py`` and again
at runtime by ``detect()``: every span has ``method="rule"`` and exact
``start``/``end`` offsets, and the function is synchronous (``detect()`` runs it
in a worker thread).
"""

from __future__ import annotations

from sanitise.rules import rule_detect

__all__ = ["rule_detect"]
