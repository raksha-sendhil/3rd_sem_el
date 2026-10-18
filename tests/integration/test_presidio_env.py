"""Environment check for the rule-pass stack: Presidio + spaCy en_core_web_lg.

These do not test Person A's recognizers - they confirm the libraries and the
model are installed and behave as expected, so a failure in ``rule_detect``
can be told apart from a broken install. Loading the model takes tens of
seconds, so they are marked ``presidio`` and deselected by default:

    pytest -m presidio
"""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = [
    pytest.mark.presidio,
    pytest.mark.skipif(
        not (
            importlib.util.find_spec("presidio_analyzer")
            and importlib.util.find_spec("en_core_web_lg")
        ),
        reason="presidio-analyzer or en_core_web_lg not installed",
    ),
]

TEXT = (
    "My name is John Smith, I am 32 years old. Call me on (212) 555-0198 "
    "or email john.smith@example.com. I live in Boston."
)


@pytest.fixture(scope="module")
def analyzer():
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        }
    )
    return AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])


def test_spacy_model_loads() -> None:
    import spacy

    nlp = spacy.load("en_core_web_lg")
    assert {"PERSON", "GPE"} <= {ent.label_ for ent in nlp(TEXT).ents}


def test_presidio_finds_explicit_identifiers(analyzer) -> None:
    found = {
        (r.entity_type, TEXT[r.start : r.end]) for r in analyzer.analyze(text=TEXT, language="en")
    }
    assert ("PERSON", "John Smith") in found
    assert ("EMAIL_ADDRESS", "john.smith@example.com") in found
    assert ("PHONE_NUMBER", "(212) 555-0198") in found
    assert ("LOCATION", "Boston") in found
