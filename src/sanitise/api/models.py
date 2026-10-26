"""Request and response bodies for the HTTP API (spec 8.4).

Responses reuse the frozen models from ``sanitise.schemas`` directly.
``doc_id`` is optional in requests; when omitted it is derived from the text
exactly as the document loader does (first 16 hex chars of SHA-256).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from sanitise.schemas import DetectorOutput, Pipeline, SanitisedRecord, SensitiveSpan


class DetectOptions(BaseModel):
    llm: bool = Field(
        default=True, description="Run the LLM pass. False = rule pass only, meta.model='none'."
    )
    model: str | None = Field(
        default=None, description="Override the Ollama model for this request."
    )
    retries: int | None = Field(
        default=None, ge=0, le=10, description="Override the repair-loop retry budget."
    )


class DetectRequest(BaseModel):
    doc_id: str | None = None
    text: str
    options: DetectOptions = Field(default_factory=DetectOptions)


class DetectRulesRequest(BaseModel):
    doc_id: str | None = None
    text: str


class DetectLLMRequest(BaseModel):
    doc_id: str | None = None
    text: str
    explicit: list[SensitiveSpan] = Field(default_factory=list)


class ScoreRequest(BaseModel):
    pred: list[DetectorOutput]
    gold: list[DetectorOutput]


# ------------------------------------------------------------------ Phase 3a (plan 6.4)


class SanitiseRequest(BaseModel):
    doc_id: str | None = None
    text: str


class SessionCreated(BaseModel):
    session_id: str


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    pipeline: Pipeline
    session_id: str
    k: int | None = Field(default=None, ge=1, le=20)


class CompareRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: str
    k: int | None = Field(default=None, ge=1, le=20)


class AttackRunRequest(BaseModel):
    session_id: str | None = None


class DocumentSummary(BaseModel):
    doc_id: str
    preview: str  # first 120 characters of the SANITISED text
    n_changes: int  # decisions whose action is not "keep"


class DocumentView(BaseModel):
    doc_id: str
    original: str
    record: SanitisedRecord


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ollama_reachable: bool
    model: str
    pipeline_version: str
