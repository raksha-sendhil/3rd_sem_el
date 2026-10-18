from __future__ import annotations

import json
from pathlib import Path

import pytest

from sanitise.errors import OffsetOutOfRange, SchemaValidationFailed
from sanitise.results import (
    GOLD_RATIONALE,
    load_gold,
    load_prediction,
    load_predictions,
    save_result,
)
from sanitise.schemas import DetectorMeta, DetectorOutput, DetectorResult, SensitiveSpan


def test_spec_gold_example_loads(tmp_path: Path) -> None:
    text = "My son has been experiencing left-sided abdominal pain. He is 32 years old."
    start = text.index("32 years old")
    (tmp_path / "case.json").write_text(
        json.dumps(
            {
                "doc_id": "a3f2c91b4e7d8a65",
                "text": text,
                "spans": [
                    {"text": "my son", "category": "RELATION", "start": 0, "end": 6},
                    {"text": "32 years old", "category": "AGE", "start": start, "end": start + 12},
                    {
                        "text": "rare occupation combined with a specific condition",
                        "category": "QUASI_COMBO",
                        "start": None,
                        "end": None,
                    },
                ],
                "annotator": "D",
                "reviewed_by": "A",
            }
        ),
        encoding="utf-8",
    )
    [gold] = load_gold(tmp_path)
    assert gold.doc_id == "a3f2c91b4e7d8a65"
    assert [s.method for s in gold.spans] == ["rule", "rule", "llm"]
    assert gold.spans[2].rationale == GOLD_RATIONALE


def test_gold_bad_offsets_reported_with_filename(tmp_path: Path) -> None:
    (tmp_path / "case_09.json").write_text(
        json.dumps(
            {
                "doc_id": "x",
                "text": "short",
                "spans": [{"text": "short", "category": "NAME", "start": 0, "end": 50}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(OffsetOutOfRange, match=r"case_09\.json"):
        load_gold(tmp_path)
    assert len(load_gold(tmp_path, check_offsets=False)[0].spans) == 1


def test_gold_bad_category(tmp_path: Path) -> None:
    (tmp_path / "case.json").write_text(
        json.dumps(
            {
                "doc_id": "x",
                "text": "flu",
                "spans": [{"text": "flu", "category": "DIAGNOSIS", "start": 0, "end": 3}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SchemaValidationFailed, match="category"):
        load_gold(tmp_path)


def test_prediction_round_trip(tmp_path: Path) -> None:
    span = SensitiveSpan(text="Bob", category="NAME", start=0, end=3, method="rule")
    result = DetectorResult(
        doc_id="abc",
        original_sha256="0" * 64,
        output=DetectorOutput(doc_id="abc", spans=[span]),
        meta=DetectorMeta(
            model="m",
            rule_ms=1,
            llm_ms=0,
            total_ms=1,
            retries=0,
            rule_span_count=1,
            llm_span_count=0,
        ),
    )
    path = save_result(result, tmp_path / "out")
    assert path.name == "abc.json"
    assert load_prediction(path) == result.output
    # A bare DetectorOutput is accepted too.
    (tmp_path / "out" / "bare.json").write_text(result.output.model_dump_json(), encoding="utf-8")
    assert len(load_predictions(tmp_path / "out")) == 2
