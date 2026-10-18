from __future__ import annotations

import hashlib

from sanitise.errors import (
    DocTooLong,
    ErrorCode,
    LLMTimeout,
    LLMUnreachable,
    OffsetOutOfRange,
    SchemaValidationFailed,
)


def test_envelope_matches_spec_shape() -> None:
    exc = SchemaValidationFailed(
        "quasi_identifiers[1].category: 'DIAGNOSIS' is not a valid PIICategory",
        stage="llm_pass",
        attempts=3,
        raw_output='{"bad": true}',
    )
    body = exc.to_envelope().model_dump(mode="json")
    assert body == {
        "error": {
            "code": "SCHEMA_VALIDATION_FAILED",
            "stage": "llm_pass",
            "attempts": 3,
            "detail": "quasi_identifiers[1].category: 'DIAGNOSIS' is not a valid PIICategory",
            "raw_output_sha256": hashlib.sha256(b'{"bad": true}').hexdigest(),
        }
    }


def test_codes_and_statuses() -> None:
    assert {c.value for c in ErrorCode} == {
        "SCHEMA_VALIDATION_FAILED",
        "LLM_TIMEOUT",
        "LLM_UNREACHABLE",
        "DOC_TOO_LONG",
        "OFFSET_OUT_OF_RANGE",
    }
    assert (DocTooLong("x").code, DocTooLong.http_status) == (ErrorCode.DOC_TOO_LONG, 413)
    assert (LLMTimeout("x").code, LLMTimeout.http_status) == (ErrorCode.LLM_TIMEOUT, 504)
    assert (LLMUnreachable("x").code, LLMUnreachable.http_status) == (
        ErrorCode.LLM_UNREACHABLE,
        503,
    )
    assert OffsetOutOfRange("x").code is ErrorCode.OFFSET_OUT_OF_RANGE


def test_with_stage_does_not_overwrite() -> None:
    exc = LLMTimeout("slow", stage="llm_pass")
    assert exc.with_stage("merge").stage == "llm_pass"
    assert LLMTimeout("slow").with_stage("merge").stage == "merge"
    assert str(exc) == "LLM_TIMEOUT [llm_pass]: slow"
