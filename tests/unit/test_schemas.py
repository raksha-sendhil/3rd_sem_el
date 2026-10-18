from __future__ import annotations

import pytest
from pydantic import ValidationError

from sanitise.schemas import (
    PIPELINE_VERSION,
    DetectorMeta,
    DetectorOutput,
    DetectorResult,
    PIICategory,
    QuasiList,
    SensitiveSpan,
)

FROZEN_CATEGORIES = [
    "NAME",
    "AGE",
    "DATE",
    "PHONE",
    "EMAIL",
    "ID_NUMBER",
    "ADDRESS",
    "LOCATION",
    "OCCUPATION",
    "RELATION",
    "PHYSICAL",
    "MEDICAL_CONDITION",
    "QUASI_COMBO",
    "OTHER",
]


def test_category_list_is_frozen() -> None:
    # Changing this list invalidates every gold annotation - see spec section 9.
    assert [c.value for c in PIICategory] == FROZEN_CATEGORIES


def test_rule_span_round_trips() -> None:
    span = SensitiveSpan(text="32 years old", category="AGE", start=182, end=194, method="rule")
    assert span.category is PIICategory.AGE
    assert span.confidence == 1.0
    assert SensitiveSpan.model_validate_json(span.model_dump_json()) == span


def test_llm_span_without_offsets_needs_rationale() -> None:
    SensitiveSpan(text="x + y", category="QUASI_COMBO", method="llm", rationale="rare pair")
    with pytest.raises(ValidationError, match="rationale is required"):
        SensitiveSpan(text="x + y", category="QUASI_COMBO", method="llm")
    with pytest.raises(ValidationError, match="rationale is required"):
        SensitiveSpan(text="x + y", category="QUASI_COMBO", method="llm", rationale="  ")


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (0, None, "both be set"),
        (None, 4, "both be set"),
        (-1, 3, ">= 0"),
        (5, 5, "greater than"),
        (6, 2, "greater than"),
    ],
)
def test_bad_offsets_rejected(start: int | None, end: int | None, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        SensitiveSpan(text="abc", category="NAME", start=start, end=end, method="rule")


def test_unknown_category_rejected() -> None:
    with pytest.raises(ValidationError):
        SensitiveSpan(text="flu", category="DIAGNOSIS", start=0, end=3, method="rule")


def test_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        SensitiveSpan(text="a", category="NAME", start=0, end=1, method="rule", confidence=1.5)


def test_detector_result_defaults_pipeline_version() -> None:
    meta = DetectorMeta(
        model="m", rule_ms=1, llm_ms=2, total_ms=3, retries=0, rule_span_count=0, llm_span_count=0
    )
    result = DetectorResult(
        doc_id="d", original_sha256="0" * 64, output=DetectorOutput(doc_id="d", spans=[]), meta=meta
    )
    assert result.meta.pipeline_version == PIPELINE_VERSION


def test_quasi_list_schema_constrains_category_to_enum() -> None:
    schema = QuasiList.model_json_schema()
    enum = schema["$defs"]["PIICategory"]["enum"]
    assert enum == FROZEN_CATEGORIES
