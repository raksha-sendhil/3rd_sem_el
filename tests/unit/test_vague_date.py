"""The vague-date filter must not hang on long digit runs (account and phone numbers)."""

import time

import pytest

from sanitise.rules import engine as sanitise_engine
from sanitise_legal.rules import engine as sanitise_legal_engine


@pytest.mark.parametrize("engine", [sanitise_engine, sanitise_legal_engine])
def test_vague_date_filter_is_fast(engine) -> None:
    start = time.perf_counter()
    for text in ("2" * 400 + "Z", "1-" * 200 + "Z", "12 " * 200 + "Z", "12.34" * 80 + "Z"):
        assert not engine._VAGUE_DATE.match(text)
    for vague in ("for 2 days", "2-3 weeks", "1.5 hours", "the last 12 months"):
        assert engine._VAGUE_DATE.match(vague)
    assert time.perf_counter() - start < 5
