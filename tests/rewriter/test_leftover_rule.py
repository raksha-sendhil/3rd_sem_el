"""A word of a removed name is a leftover only when the document uses it nowhere else."""

from __future__ import annotations

from sanitise.agents.rewriter import prompt
from sanitise.schemas import Action, PIICategory, RoutingDecision

TEXT = ("Seen at North Valley Imaging Center.\nDiagnostic Imaging Report - Chest X-ray.\n"
        "Lives near Valley Road.")  # fmt: skip


def _clinic() -> RoutingDecision:
    start = TEXT.index("North Valley Imaging Center")
    return RoutingDecision(
        span_text="North Valley Imaging Center", category=PIICategory.LOCATION,
        action=Action.GENERALISE, replacement="a place", reason="test", start=start,
        end=start + len("North Valley Imaging Center"),
    )  # fmt: skip


def test_word_used_elsewhere_in_the_document_is_not_a_leftover() -> None:
    token = prompt.SOURCE_TEXT.set(TEXT)
    try:
        value = prompt.safe_value("Diagnostic Imaging Report - Chest X-ray (2 views)", [_clinic()])
    finally:
        prompt.SOURCE_TEXT.reset(token)
    assert "Imaging Report" in value


def test_word_found_only_inside_the_removed_name_still_counts() -> None:
    only_here = "Seen at North Valley Imaging Center.\nChest X-ray."
    d = _clinic()
    token = prompt.SOURCE_TEXT.set(only_here)
    try:
        value = prompt.safe_value("referred by the Imaging team", [d])
    finally:
        prompt.SOURCE_TEXT.reset(token)
    assert "Imaging" not in value
