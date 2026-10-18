from __future__ import annotations

import pytest
from pydantic import ValidationError

from sanitise.llm.base import format_validation_error, inline_refs, parse_json_lenient
from sanitise.llm.mock import fabricate
from sanitise.schemas import DetectorResult, QuasiList


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Sure! Here is the JSON:\n{"a": 1}\nHope that helps.',
    ],
)
def test_parse_json_lenient(raw: str) -> None:
    assert parse_json_lenient(raw) == {"a": 1}


def test_parse_json_lenient_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_json_lenient("I cannot help with that.")


def test_format_validation_error_paths() -> None:
    with pytest.raises(ValidationError) as info:
        QuasiList.model_validate(
            {
                "quasi_identifiers": [
                    {"text": "a", "category": "NAME", "rationale": "r"},
                    {"text": "b", "category": "DIAGNOSIS", "rationale": "r"},
                    {"text": "c", "category": "AGE"},
                ]
            }
        )
    detail = format_validation_error(info.value)
    assert "quasi_identifiers[1].category" in detail and "'DIAGNOSIS'" in detail
    assert "quasi_identifiers[2].rationale: Field required" in detail


def test_inline_refs_removes_defs() -> None:
    flat = inline_refs(QuasiList.model_json_schema())
    assert "$defs" not in flat and "$ref" not in str(flat)
    item = flat["properties"]["quasi_identifiers"]["items"]
    assert "NAME" in item["properties"]["category"]["enum"]


@pytest.mark.parametrize("model", [QuasiList, DetectorResult])
def test_fabricate_produces_valid_instances(model: type) -> None:
    model.model_validate(fabricate(inline_refs(model.model_json_schema())))
