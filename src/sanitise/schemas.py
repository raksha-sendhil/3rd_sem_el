"""Shared schemas - the API contract between all four tracks (spec section 8.1).

FROZEN ON DAY 1. Every annotation, prompt example and test depends on these
definitions. Do not add, rename or remove a field or an enum member without a
whole-team decision; changing ``PIICategory`` mid-week invalidates the gold set.

Owner: Person C (Platform and Integration).

Beyond the literal spec listing this module also holds the models that cross a
seam between two people (``QuasiHit``/``QuasiList`` from section 8.2 and
``SpanMetrics`` from section 8.2), so every boundary is validated by one
definition. The only behaviour added to the spec models is validation that the
spec states in prose (offsets, the llm-rationale rule).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

PIPELINE_VERSION = "0.1.0"


class PIICategory(str, Enum):
    NAME = "NAME"
    AGE = "AGE"
    DATE = "DATE"
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    ID_NUMBER = "ID_NUMBER"
    ADDRESS = "ADDRESS"
    LOCATION = "LOCATION"
    OCCUPATION = "OCCUPATION"
    RELATION = "RELATION"
    PHYSICAL = "PHYSICAL"  # height, weight, build
    MEDICAL_CONDITION = "MEDICAL_CONDITION"
    SYMPTOM = "SYMPTOM"  # added 22 Oct (decision D21): a symptom a person reports
    QUASI_COMBO = "QUASI_COMBO"  # multi-attribute linkage
    OTHER = "OTHER"


# ================= AGENT 1 : built in Phase 2 =================


class SensitiveSpan(BaseModel):
    text: str
    category: PIICategory
    start: int | None = None  # None for LLM-inferred combinations
    end: int | None = None
    method: Literal["rule", "llm"]
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    rationale: str | None = None  # required when method == "llm"

    @model_validator(mode="after")
    def _check_offsets_and_rationale(self) -> SensitiveSpan:
        if (self.start is None) != (self.end is None):
            raise ValueError("start and end must both be set or both be None")
        if self.start is not None and self.end is not None:
            if self.start < 0:
                raise ValueError(f"start must be >= 0, got {self.start}")
            if self.end <= self.start:
                raise ValueError(f"end ({self.end}) must be greater than start ({self.start})")
        if self.method == "llm" and not (self.rationale and self.rationale.strip()):
            raise ValueError("rationale is required when method == 'llm'")
        return self

    @property
    def has_offsets(self) -> bool:
        return self.start is not None


class DetectorOutput(BaseModel):
    doc_id: str
    spans: list[SensitiveSpan]


class DetectorMeta(BaseModel):
    model: str
    pipeline_version: str = PIPELINE_VERSION
    rule_ms: int
    llm_ms: int
    total_ms: int
    retries: int
    rule_span_count: int
    llm_span_count: int


class DetectorResult(BaseModel):
    doc_id: str
    original_sha256: str
    output: DetectorOutput
    meta: DetectorMeta


# ================= Seams between people (spec 8.2) =================


class QuasiHit(BaseModel):
    """One quasi-identifier as returned by the model (Person B's prompt)."""

    text: str
    category: PIICategory
    rationale: str


class QuasiList(BaseModel):
    """The schema handed to Ollama for constrained decoding in the LLM pass."""

    quasi_identifiers: list[QuasiHit]


class SpanMetrics(BaseModel):
    """Scoring result (Person D). Also the response body of POST /v1/score."""

    precision: float
    recall: float
    f1: float
    support: int
    per_category: dict[str, dict[str, float]]


# ================= AGENTS 2 and 3 : Phase 3, defined now =================


class Importance(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Action(str, Enum):
    PLACEHOLDER = "placeholder"
    GENERALISE = "generalise"
    KEEP = "keep"


class FactTuple(BaseModel):
    subject: str
    relation: str
    value: str
    importance: Importance


class ExtractorOutput(BaseModel):
    doc_id: str
    facts: list[FactTuple]


class RoutingDecision(BaseModel):
    span_text: str
    category: PIICategory
    action: Action
    replacement: str | None = None
    reason: str


class RewriterOutput(BaseModel):
    doc_id: str
    safe_text: str
    decisions: list[RoutingDecision]
    placeholders: dict[str, str]


__all__ = [
    "PIPELINE_VERSION",
    "Action",
    "DetectorMeta",
    "DetectorOutput",
    "DetectorResult",
    "ExtractorOutput",
    "FactTuple",
    "Importance",
    "PIICategory",
    "QuasiHit",
    "QuasiList",
    "RewriterOutput",
    "RoutingDecision",
    "SensitiveSpan",
    "SpanMetrics",
]
