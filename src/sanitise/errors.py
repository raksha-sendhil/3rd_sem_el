"""Pipeline errors and the error envelope (spec section 8.5).

Every failure that crosses a public boundary (CLI, HTTP) is a ``PipelineError``
carrying one of the five frozen codes. The HTTP layer serialises it with
``to_envelope()``; the CLI prints the same envelope.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Literal

from pydantic import BaseModel

Stage = Literal["load", "rule_pass", "llm_pass", "merge", "validation", "score"]


class ErrorCode(str, Enum):
    SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_UNREACHABLE = "LLM_UNREACHABLE"
    DOC_TOO_LONG = "DOC_TOO_LONG"
    OFFSET_OUT_OF_RANGE = "OFFSET_OUT_OF_RANGE"


class ErrorBody(BaseModel):
    code: ErrorCode
    stage: str | None = None
    attempts: int | None = None
    detail: str
    raw_output_sha256: str | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody


class PipelineError(Exception):
    """Base class. ``stage`` may be filled in later by the orchestrator."""

    code: ErrorCode = ErrorCode.SCHEMA_VALIDATION_FAILED
    http_status: int = 500

    def __init__(
        self,
        detail: str,
        *,
        stage: Stage | None = None,
        attempts: int | None = None,
        raw_output: str | None = None,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.stage = stage
        self.attempts = attempts
        self.raw_output_sha256 = (
            hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
            if raw_output is not None
            else None
        )

    def with_stage(self, stage: Stage) -> PipelineError:
        if self.stage is None:
            self.stage = stage
        return self

    def to_envelope(self) -> ErrorEnvelope:
        return ErrorEnvelope(
            error=ErrorBody(
                code=self.code,
                stage=self.stage,
                attempts=self.attempts,
                detail=self.detail,
                raw_output_sha256=self.raw_output_sha256,
            )
        )

    def __str__(self) -> str:
        where = f" [{self.stage}]" if self.stage else ""
        return f"{self.code.value}{where}: {self.detail}"


class SchemaValidationFailed(PipelineError):
    """The model never produced output that validates, even after the repair loop."""

    code = ErrorCode.SCHEMA_VALIDATION_FAILED
    http_status = 502


class LLMTimeout(PipelineError):
    code = ErrorCode.LLM_TIMEOUT
    http_status = 504


class LLMUnreachable(PipelineError):
    code = ErrorCode.LLM_UNREACHABLE
    http_status = 503


class DocTooLong(PipelineError):
    code = ErrorCode.DOC_TOO_LONG
    http_status = 413


class OffsetOutOfRange(PipelineError):
    """A span's offsets fall outside the document or do not match its text."""

    code = ErrorCode.OFFSET_OUT_OF_RANGE
    http_status = 500
