"""
schemas.py - COPY of the frozen contract from spec section 8.1.

Person C owns the real file. This copy exists so the evaluation code and tests
run standalone. On integration day, delete this file and use C's version.
If the two ever differ, C's version wins and this copy must be re-synced.
"""
from enum import Enum
from typing import Literal, Protocol

from pydantic import BaseModel, Field

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
