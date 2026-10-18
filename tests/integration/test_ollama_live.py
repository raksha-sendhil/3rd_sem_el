"""Smoke tests against a real Ollama server (Day 6). Deselected by default.

ollama pull llama3:8b
pytest -m ollama

The autouse settings fixture clears every SANITISE_* variable, so these tests
take their own overrides (useful on Colab or a slow CPU-only machine):

OLLAMA_TEST_MODEL      model to test (default llama3:8b)
OLLAMA_TEST_TIMEOUT_S  per-request timeout (default 600 - the first call also loads the model)
"""

from __future__ import annotations

import os

import pytest

from sanitise.agents.detector.pipeline import detect
from sanitise.agents.detector.quasi import infer_quasi
from sanitise.config import Settings
from sanitise.llm import OllamaClient
from sanitise.loader import compute_doc_id
from sanitise.schemas import PIICategory as C
from sanitise.schemas import QuasiList
from tests.conftest import SAMPLE_TEXT, fake_rule_detect

pytestmark = pytest.mark.ollama


@pytest.fixture
async def ollama() -> OllamaClient:
    settings = Settings(
        llm_backend="ollama",
        model=os.environ.get("OLLAMA_TEST_MODEL", "llama3:8b"),
        llm_timeout_s=float(os.environ.get("OLLAMA_TEST_TIMEOUT_S", "600")),
    )
    client = OllamaClient.from_settings(settings)
    status = await client.health()
    if not status["model_available"]:
        await client.aclose()
        pytest.skip(status["detail"] or "Ollama not available")
    yield client
    await client.aclose()


async def test_constrained_json_call(ollama: OllamaClient) -> None:
    result = await ollama.json_call(
        "List quasi-identifiers in the document as JSON.", SAMPLE_TEXT, QuasiList
    )
    assert isinstance(result, QuasiList)


async def test_detect_end_to_end(ollama: OllamaClient) -> None:
    result = await detect(compute_doc_id(SAMPLE_TEXT), SAMPLE_TEXT, ollama)
    assert result.meta.model == ollama.model
    assert result.meta.llm_ms > 0


async def test_infer_quasi_live(ollama: OllamaClient) -> None:
    """Person B's pass on a real model: allowed categories only, grounded offsets fit."""
    spans = await infer_quasi(SAMPLE_TEXT, fake_rule_detect(SAMPLE_TEXT), ollama)
    allowed = {C.MEDICAL_CONDITION, C.OCCUPATION, C.LOCATION, C.QUASI_COMBO}
    for span in spans:
        assert span.category in allowed and span.method == "llm" and span.rationale
        if span.start is not None:
            assert SAMPLE_TEXT[span.start : span.end] == span.text
        else:
            assert span.category is C.QUASI_COMBO
