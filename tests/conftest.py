"""Shared test fixtures.

Nothing here needs Ollama, Presidio or spaCy: the LLM is ``MockLLM`` and the
rule pass is a fake built from known substrings of ``SAMPLE_TEXT``. Tests that
need a live model are marked ``@pytest.mark.ollama`` and deselected by default
(run them with ``pytest -m ollama``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from sanitise.agents.detector import pipeline
from sanitise.config import Settings, get_settings
from sanitise.llm.mock import MockLLM
from sanitise.loader import compute_doc_id
from sanitise.schemas import PIICategory, QuasiHit, QuasiList, SensitiveSpan

SAMPLE_TEXT = (
    "My son has been experiencing left-sided abdominal pain for about ten weeks. "
    "He is 32 years old, 6 ft 3 inches tall, and weighs approximately 220 pounds. "
    "He works as a lighthouse keeper in Port Isaac. You can reach me on 07700 900123."
)

# (substring, category) pairs the fake rule pass "finds" in SAMPLE_TEXT.
SAMPLE_RULE_HITS: list[tuple[str, PIICategory]] = [
    ("My son", PIICategory.RELATION),
    ("32 years old", PIICategory.AGE),
    ("6 ft 3 inches", PIICategory.PHYSICAL),
    ("approximately 220 pounds", PIICategory.PHYSICAL),
    ("07700 900123", PIICategory.PHONE),
]

SAMPLE_QUASI = QuasiList(
    quasi_identifiers=[
        QuasiHit(
            text="lighthouse keeper + Port Isaac",
            category=PIICategory.QUASI_COMBO,
            rationale="A rare occupation in a small village identifies one person.",
        ),
        QuasiHit(
            text="about ten weeks of left-sided abdominal pain",
            category=PIICategory.QUASI_COMBO,
            rationale="A precise symptom timeline narrows the candidate pool.",
        ),
    ]
)


def span_for(text: str, needle: str, category: PIICategory) -> SensitiveSpan:
    start = text.index(needle)
    return SensitiveSpan(
        text=needle, category=category, start=start, end=start + len(needle), method="rule"
    )


def fake_rule_detect(text: str) -> list[SensitiveSpan]:
    return [span_for(text, needle, cat) for needle, cat in SAMPLE_RULE_HITS if needle in text]


@pytest.fixture(autouse=True)
def _isolated_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[None]:
    """Each test gets default settings with the mock backend, unaffected by a developer's .env."""
    for key in list(os.environ):
        if key.startswith("SANITISE_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))  # no stray .env is picked up
    monkeypatch.setenv("SANITISE_LLM_BACKEND", "mock")
    monkeypatch.setenv("SANITISE_TRANSPORT_RETRIES", "0")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    return get_settings()


@pytest.fixture
def sample_text() -> str:
    return SAMPLE_TEXT


@pytest.fixture
def sample_doc_id() -> str:
    return compute_doc_id(SAMPLE_TEXT)


@pytest.fixture
def mock_llm() -> MockLLM:
    """A MockLLM that answers every QuasiList call with ``SAMPLE_QUASI``."""
    return MockLLM(handler=lambda call: SAMPLE_QUASI)


@pytest.fixture
def fake_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap Person A's rule pass for the fake, so the pipeline has explicit spans to work with."""
    monkeypatch.setattr(pipeline.rules, "rule_detect", fake_rule_detect)


@pytest.fixture
def fixture_file(tmp_path: Path) -> Path:
    path = tmp_path / "case_01.json"
    path.write_text(json.dumps({"text": SAMPLE_TEXT}), encoding="utf-8")
    return path


@pytest.fixture
def gold_dir(tmp_path: Path) -> Path:
    """A one-document gold set in the spec 8.3 format."""
    directory = tmp_path / "gold"
    directory.mkdir()
    spans = [
        {"text": s.text, "category": s.category.value, "start": s.start, "end": s.end}
        for s in fake_rule_detect(SAMPLE_TEXT)
    ]
    spans.append(
        {
            "text": "rare occupation in a small village",
            "category": "QUASI_COMBO",
            "start": None,
            "end": None,
        }
    )
    (directory / "case_01.json").write_text(
        json.dumps(
            {
                "doc_id": compute_doc_id(SAMPLE_TEXT),
                "text": SAMPLE_TEXT,
                "spans": spans,
                "annotator": "D",
                "reviewed_by": "A",
            }
        ),
        encoding="utf-8",
    )
    return directory
