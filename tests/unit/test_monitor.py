"""Session privacy monitor: attributes, the hashed ledger, the pool, thresholds and the router."""

from __future__ import annotations

import pytest

from sanitise.agents.rewriter.ontology import generalise
from sanitise.config import Settings
from sanitise.monitor import AllowAllMonitor, DisclosureLedger, LedgerMonitor, build_monitor
from sanitise.monitor.attributes import mentioned, record_attributes, replace_form
from sanitise.rag.store import MemoryDocumentStore
from sanitise.schemas import (
    Action,
    DetectorOutput,
    ExtractorOutput,
    FactTuple,
    Importance,
    PIICategory,
    RetrievalHit,
    RewriterOutput,
    RoutingDecision,
    SanitisedRecord,
    SanitiseMeta,
    Subject,
)

C = PIICategory

# (age, town, occupation, condition): eleven patients share vertigo; the target is p1
PATIENTS = {
    "p1": ("36 years old", "Coonoor", "neurosurgeon", "vertigo"),
    "p2": ("38 years old", "Ooty", "nurse", "vertigo"),
    "p3": ("34 years old", "Kodaikanal", "teacher", "vertigo"),
    "p4": ("52 years old", "Coonoor", "teacher", "vertigo"),
    "p5": ("31 years old", "Chennai", "nurse", "vertigo"),
    "p6": ("33 years old", "Madurai", "teacher", "vertigo"),
    "p7": ("57 years old", "Salem", "nurse", "vertigo"),
    "p8": ("45 years old", "Erode", "teacher", "vertigo"),
    "p9": ("29 years old", "Chennai", "teacher", "acne"),
    "p10": ("61 years old", "Madurai", "nurse", "acne"),
    "p11": ("35 years old", "Ooty", "nurse", "vertigo"),
    "p12": ("37 years old", "Kodaikanal", "nurse", "vertigo"),
    "p13": ("32 years old", "Coimbatore", "nurse", "vertigo"),
}


def make_record(doc_id: str, age: str, town: str, job: str, condition: str) -> SanitisedRecord:
    decisions = [
        RoutingDecision(span_text="Priya", category=C.NAME, action=Action.PLACEHOLDER,
                        replacement="[PERSON_1]", reason="name", start=0, end=5),
        RoutingDecision(span_text=age, category=C.AGE, action=Action.GENERALISE,
                        replacement=generalise(age, C.AGE), reason="age", start=6, end=8),
        RoutingDecision(span_text=town, category=C.LOCATION, action=Action.GENERALISE,
                        replacement=generalise(town, C.LOCATION), reason="town", start=9, end=12),
        RoutingDecision(span_text=job, category=C.OCCUPATION, action=Action.GENERALISE,
                        replacement=generalise(job, C.OCCUPATION), reason="job", start=13, end=16),
        RoutingDecision(span_text=condition, category=C.MEDICAL_CONDITION, action=Action.KEEP,
                        reason="common", start=17, end=20),
    ]  # fmt: skip
    facts = [
        FactTuple(subject="patient", relation=r, value=v, importance=Importance.HIGH)
        for r, v in [
            ("age", age),
            ("lives_in", town),
            ("works_as", job),
            ("has_condition", condition),
        ]
    ]
    safe = ". ".join(d.replacement or d.span_text for d in decisions)
    return SanitisedRecord(
        doc_id=doc_id,
        original_sha256="0" * 64,
        subjects=[Subject(id="patient", label="the patient")],
        detector=DetectorOutput(doc_id=doc_id, spans=[]),
        extractor=ExtractorOutput(doc_id=doc_id, facts=facts),
        rewriter=RewriterOutput(doc_id=doc_id, safe_text=safe, decisions=decisions,
                                placeholders={}),
        meta=SanitiseMeta(model="test", detect_ms=0, extract_ms=0, rewrite_ms=0, total_ms=0,
                          retries=0, leak_retries=0),
    )  # fmt: skip


@pytest.fixture
def store() -> MemoryDocumentStore:
    s = MemoryDocumentStore()
    for doc_id, values in PATIENTS.items():
        s.put(doc_id, "original", make_record(doc_id, *values))
    return s


@pytest.fixture
def monitor(store: MemoryDocumentStore) -> LedgerMonitor:
    return LedgerMonitor(store, k=5, max_attributes=3, ledger=DisclosureLedger(salt="test"))


HIT = [RetrievalHit(doc_id="p1", score=1.0)]


# ------------------------------------------------------------------ attributes


def test_record_attributes_follow_the_ladder(store: MemoryDocumentStore) -> None:
    attrs = {a.type: a for a in record_attributes(store.sanitised("p1"))}
    assert set(attrs) == {"age", "location", "occupation", "condition"}  # the name is a placeholder
    assert attrs["location"].ladder == ("a town in the Nilgiris district", "a town in Tamil Nadu")
    assert attrs["age"].ladder[:2] == ("in their 30s", "a young adult")
    assert attrs["condition"].shown == "vertigo"  # kept as it is
    assert all(a.subject_id == "p1:patient" for a in attrs.values())


def test_mentioned_and_replaced_by_core() -> None:
    assert mentioned("She lives in the Nilgiris district.", "a town in the Nilgiris district")
    assert mentioned("A patient in her thirties.", "in their 30s")
    assert not mentioned("A patient in her forties.", "in their 30s")
    assert not mentioned("Vertigos are common", "vertigo")  # whole words only
    text = replace_form("She lives in the Nilgiris district.", "a town in the Nilgiris district",
                        "a town in Tamil Nadu")  # fmt: skip
    assert text == "She lives in a town in Tamil Nadu."


# ------------------------------------------------------------------ ledger


def test_ledger_keeps_only_keyed_hashes(tmp_path) -> None:
    path = tmp_path / "ledger.sqlite"
    ledger = DisclosureLedger(path, salt="k1")
    h = ledger.hash("condition", "Vertigo ")
    assert (
        h
        == ledger.hash("condition", "vertigo")
        != DisclosureLedger(salt="k2").hash("condition", "vertigo")
    )
    ledger.record("s1", 1, [("p1:patient", "condition", h)])
    ledger.record("s1", 2, [("p1:patient", "condition", h)])  # a repeat keeps round 1
    assert ledger.known("s1") == {"p1:patient": {"condition": {h}}}
    assert [d.round for d in ledger.entries("s1")] == [1]
    assert b"vertigo" not in path.read_bytes()
    ledger.close()
    assert DisclosureLedger(path, salt="k1").known("s1")  # it persists
    ledger = DisclosureLedger(path, salt="k1")
    ledger.forget("s1")
    assert ledger.known("s1") == {}


# ------------------------------------------------------------------ the monitor


def test_pool_counts_every_consistent_record(monitor: LedgerMonitor) -> None:
    h = monitor.ledger.hash
    assert monitor.pool({"condition": {h("condition", "vertigo")}}) == 11
    assert monitor.pool({"condition": {h("condition", "vertigo")},
                         "age": {h("age", "in their 30s")}}) == 8  # fmt: skip
    # a coarser form matches everyone whose ladder reaches it
    assert monitor.pool({"location": {h("location", "a town in the Nilgiris district")}}) == 4
    assert monitor.pool({"location": {h("location", "a town in Tamil Nadu")}}) == 6


def test_answer_without_personal_details_is_allowed(monitor: LedgerMonitor) -> None:
    v = monitor.check(
        "s", "q", HIT, "Vertigo is usually treated with rest and fluids... [PERSON_1]"
    )
    assert v.verdict == "allowed"  # "Vertigo" here is the kept condition
    v = monitor.check("s2", "q", HIT, "Drink plenty of water.")
    assert v.verdict == "allowed" and v.candidate_pool is None


def test_stepwise_attack_is_generalised_then_blocked(monitor: LedgerMonitor) -> None:
    # round 1: the condition alone, 11 records match
    v1 = monitor.check("s", "q1", HIT, "Yes, one patient has vertigo.")
    assert (v1.verdict, v1.candidate_pool, v1.disclosed) == ("allowed", 11, ["condition"])
    # round 2: + age band, 8 match
    v2 = monitor.check("s", "q2", HIT, "The patient with vertigo is in their 30s.")
    assert (v2.verdict, v2.candidate_pool) == ("allowed", 8)
    # round 3: + occupation, 6 match: still allowed (k = 5), the third kind of detail (N = 3)
    v3 = monitor.check("s", "q3", HIT, "They are a healthcare worker in their 30s with vertigo.")
    assert (v3.verdict, v3.candidate_pool) == ("allowed", 6)
    assert v3.disclosed == ["age", "condition", "occupation"]
    # round 4: a fourth kind of detail is over the limit: blocked, nothing recorded
    before = monitor.ledger.entries("s")
    v4 = monitor.check("s", "q4", HIT, "They live in a town in the Nilgiris district.")
    assert v4.verdict == "blocked" and "location" in v4.reason
    assert monitor.ledger.entries("s") == before
    assert sorted({d.attribute_type for d in before}) == ["age", "condition", "occupation"]


def test_sessions_are_independent(monitor: LedgerMonitor) -> None:
    monitor.check("a", "q", HIT, "vertigo, in their 30s")
    monitor.check("a", "q", HIT, "a healthcare worker")
    v = monitor.check("b", "q", HIT, "a patient with vertigo")
    assert (v.verdict, v.candidate_pool) == ("allowed", 11)  # session b knows only the condition


def test_a_too_specific_detail_climbs_the_ladder(monitor: LedgerMonitor) -> None:
    # 4 records in the Nilgiris (k = 5), 6 "in Tamil Nadu": one step up is enough
    v = monitor.check("s", "q", HIT, "They live in the Nilgiris district.")
    assert (v.verdict, v.candidate_pool) == ("generalised", 6)
    assert v.revised_answer == "They live in a town in Tamil Nadu."
    assert "4 -> 6" in v.reason and v.disclosed == ["location"]


def test_only_the_generic_phrase_can_be_safe(monitor: LedgerMonitor) -> None:
    # acne: 2 records; "a skin condition": 2 records; the generic phrase reveals nothing
    hit = [RetrievalHit(doc_id="p9", score=1.0)]
    v = monitor.check("s", "q", hit, "The patient has acne.")
    assert v.verdict == "generalised" and v.revised_answer == "The patient has a medical condition."
    assert v.disclosed == [] and monitor.ledger.entries("s") == []  # nothing was revealed


def test_blocked_when_generalising_is_not_allowed(store: MemoryDocumentStore) -> None:
    strict = LedgerMonitor(store, k=5, max_climb=0, ledger=DisclosureLedger(salt="t"))
    v = strict.check("s", "q", HIT, "They live in the Nilgiris district.")  # 4 match, no climbing
    assert v.verdict == "blocked" and v.candidate_pool == 4 and strict.ledger.entries("s") == []


def test_build_monitor_from_settings(store: MemoryDocumentStore) -> None:
    assert isinstance(build_monitor(Settings(monitor="off"), store), AllowAllMonitor)
    m = build_monitor(Settings(llm_backend="mock", monitor_k=4), store)
    assert isinstance(m, LedgerMonitor) and m.k == 4
