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
    # Phase 3a (docs/phase3_plan.md 6.1): offsets of span_text in the ORIGINAL text, so the
    # document viewer can highlight each change. None for QUASI_COMBO.
    start: int | None = None
    end: int | None = None


class RewriterOutput(BaseModel):
    doc_id: str
    safe_text: str
    decisions: list[RoutingDecision]
    placeholders: dict[str, str]


# ================= PHASE 3a : contract in docs/phase3_plan.md section 6 =================

# Fact relations (plan 6.3). Agent 2 output, the gold facts and A's normalisation use these.
RELATIONS: tuple[str, ...] = (
    "age",
    "sex",
    "lives_in",
    "works_as",
    "studies_at",
    "treated_at",
    "has_condition",
    "has_symptom",
    "symptom_duration",
    "took_medication",
    "had_procedure",
    "event_date",
    "relation_to",
    "height",
    "weight",
    "contact",
    "identifier",
    "other",
)

Pipeline = Literal["baseline", "protected"]
Verdict = Literal["allowed", "generalised", "blocked"]
AttackKind = Literal["direct", "linkage", "stepwise"]


class Subject(BaseModel):
    """A person a document talks about (plan 6.1). Built by Person A's ``subjects()``."""

    id: str  # "writer", "writer.son", "person_1"
    label: str  # "the writer", "the writer's son"
    mentions: list[str] = Field(default_factory=list)  # exact strings from the text


class SanitiseMeta(BaseModel):
    model: str
    pipeline_version: str = PIPELINE_VERSION
    detect_ms: int
    extract_ms: int
    rewrite_ms: int
    total_ms: int
    retries: int  # LLM schema retries across the three agents
    leak_retries: int  # rewrites redone because the leak check found a leak


class SanitisedRecord(BaseModel):
    """Everything the offline path produces for one document."""

    doc_id: str
    original_sha256: str
    subjects: list[Subject]
    detector: DetectorOutput
    extractor: ExtractorOutput
    rewriter: RewriterOutput
    residual_leaks: list[SensitiveSpan] = Field(default_factory=list)  # [] when clean
    meta: SanitiseMeta


class RetrievalHit(BaseModel):
    doc_id: str
    score: float  # higher is better
    dense_rank: int | None = None
    sparse_rank: int | None = None


class MonitorVerdict(BaseModel):
    """The session monitor's decision. The Phase 3a stub always allows."""

    verdict: Verdict = "allowed"
    candidate_pool: int | None = None  # filled by the Phase 3b monitor
    disclosed: list[str] = Field(default_factory=list)
    reason: str = "no monitor installed"


class RagAnswer(BaseModel):
    pipeline: Pipeline
    query: str
    answer: str
    doc_ids: list[str]  # retrieved documents, in rank order
    payload: str  # the literal text sent to the model (payload inspector)
    monitor: MonitorVerdict = Field(default_factory=MonitorVerdict)
    latency_ms: int


class CompareResponse(BaseModel):
    baseline: RagAnswer
    protected: RagAnswer


class RoundLeakage(BaseModel):
    disclosed: list[str]  # target attributes revealed so far: "age", "town", "name", ...
    candidate_pool: int  # people still consistent with what was revealed
    identified: bool  # pool <= 1, or a direct identifier of the target appeared


class AttackRound(BaseModel):
    round: int
    query: str
    baseline: RagAnswer
    protected: RagAnswer
    baseline_leakage: RoundLeakage | None = None  # None outside an attack (plain /v1/compare)
    protected_leakage: RoundLeakage | None = None


class AttackDefinition(BaseModel):
    """One file in data/attacks/ (plan 6.3)."""

    attack_id: str
    kind: AttackKind
    name: str
    target_person_id: str
    queries: list[str] = Field(min_length=1)


class AttackRun(BaseModel):
    attack_id: str
    session_id: str
    rounds: list[AttackRound]
    baseline_identified_at: int | None = None  # first round the target was identified
    protected_identified_at: int | None = None


class SessionTrace(BaseModel):
    session_id: str
    rounds: list[AttackRound]


class Person(BaseModel):
    """One row of data/corpus/people.json (plan 6.3). Used only by the scorers."""

    person_id: str
    name: str
    sex: str
    age: int
    age_band: str
    town: str
    district: str
    state: str
    occupation: str
    occupation_group: str
    conditions: list[str] = Field(default_factory=list)
    phone: str | None = None
    doc_ids: list[str] = Field(default_factory=list)


class GoldRoute(BaseModel):
    text: str
    category: PIICategory
    action: Action


class GoldRouting(BaseModel):
    """One file in data/gold_rewrite/: one entry per gold span, same order as data/gold/."""

    doc_id: str
    routing: list[GoldRoute]


class QAItem(BaseModel):
    """One entry of data/qa/qa.json."""

    qa_id: str
    question: str
    expected: list[str]  # every string must appear in a correct answer (case-insensitive)
    doc_ids: list[str] = Field(default_factory=list)


class FactMetrics(BaseModel):
    coverage: float  # gold facts matched / gold facts
    faithfulness: float  # predicted facts matching a gold fact / predicted facts
    high_coverage: float  # coverage over importance == high only
    support: int


class RewriteMetrics(BaseModel):
    leak_rate: float  # spans routed to placeholder/generalise still present / those spans
    utility: float  # high-importance gold fact values still in the safe text / those facts
    routing_accuracy: float  # predicted action == gold action
    support: int


class QAMetrics(BaseModel):
    baseline_accuracy: float
    protected_accuracy: float
    support: int


__all__ = [
    "PIPELINE_VERSION",
    "RELATIONS",
    "Action",
    "AttackDefinition",
    "AttackKind",
    "AttackRound",
    "AttackRun",
    "CompareResponse",
    "DetectorMeta",
    "DetectorOutput",
    "DetectorResult",
    "ExtractorOutput",
    "FactMetrics",
    "FactTuple",
    "GoldRoute",
    "GoldRouting",
    "Importance",
    "MonitorVerdict",
    "PIICategory",
    "Person",
    "Pipeline",
    "QAItem",
    "QAMetrics",
    "QuasiHit",
    "QuasiList",
    "RagAnswer",
    "RetrievalHit",
    "RewriteMetrics",
    "RewriterOutput",
    "RoundLeakage",
    "RoutingDecision",
    "SanitiseMeta",
    "SanitisedRecord",
    "SensitiveSpan",
    "SessionTrace",
    "SpanMetrics",
    "Subject",
    "Verdict",
]
