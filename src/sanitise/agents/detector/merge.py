"""Stage 6 - merge.  OWNER: Person A (Deterministic Logic).

The implementation lives in ``sanitise.rules.merge``: union, drop
quasi-identifiers that restate an explicit identifier, resolve overlaps by the
fixed category precedence order. This module is the seam ``detect()`` imports.
"""

from __future__ import annotations

from sanitise.rules import merge_spans

__all__ = ["merge_spans"]
