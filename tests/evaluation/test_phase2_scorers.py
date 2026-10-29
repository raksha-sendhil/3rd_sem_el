"""Tests for Person D's Phase 3 scorers (hand-made predictions, edge cases)."""

from __future__ import annotations

from sanitise.evaluation.attacks import score_round, target_info
from sanitise.evaluation.facts import score_facts
from sanitise.evaluation.qa import score_qa
from sanitise.evaluation.rewrite import score_rewrite
from sanitise.schemas import (
    Action,
    ExtractorOutput,
    FactTuple,
    GoldRoute,
    GoldRouting,
    Importance,
    Person,
    PIICategory,
    QAItem,
    RagAnswer,
    RewriterOutput,
    RoutingDecision,
)


def fact(subject, relation, value, imp="medium"):
    return FactTuple(subject=subject, relation=relation, value=value, importance=Importance(imp))


def out(doc_id, *facts):
    return ExtractorOutput(doc_id=doc_id, facts=list(facts))


def answer(text, pipeline="baseline", query="q"):
    return RagAnswer(
        pipeline=pipeline, query=query, answer=text, doc_ids=[], payload="", latency_ms=0
    )


# ------------------------------------------------------------------ facts


def test_facts_perfect_and_variants():
    gold = [
        out(
            "d",
            fact("writer", "age", "34 year old"),
            fact("writer", "has_condition", "vertigo", "high"),
        )
    ]
    pred = [
        out("d", fact("writer", "age", "34 years old"), fact("writer", "has_condition", "Vertigo"))
    ]
    m = score_facts(pred, gold)
    assert (m.coverage, m.faithfulness, m.high_coverage, m.support) == (1.0, 1.0, 1.0, 2)


def test_facts_wrong_subject_relation_and_partial_value():
    gold = [out("d", fact("writer.son", "has_symptom", "left-sided abdominal pain", "high"))]
    assert (
        score_facts(
            [out("d", fact("writer", "has_symptom", "left-sided abdominal pain"))], gold
        ).coverage
        == 0
    )
    assert (
        score_facts(
            [out("d", fact("writer.son", "has_condition", "left-sided abdominal pain"))], gold
        ).coverage
        == 0
    )
    # a one-word value is not enough to match a long gold value
    assert score_facts([out("d", fact("writer.son", "has_symptom", "pain"))], gold).coverage == 0
    assert (
        score_facts(
            [out("d", fact("writer.son", "has_symptom", "abdominal pain on the left"))], gold
        ).coverage
        == 1
    )


def test_facts_one_to_one_and_extras():
    gold = [
        out(
            "d",
            fact("writer", "has_symptom", "dry cough"),
            fact("writer", "has_symptom", "dry cough at night"),
        )
    ]
    pred = [out("d", fact("writer", "has_symptom", "dry cough at night"))]
    m = score_facts(pred, gold)
    assert m.coverage == 0.5 and m.faithfulness == 1.0
    extra = [out("d", fact("writer", "has_symptom", "dry cough"), fact("writer", "other", "x y"))]
    assert score_facts(extra, gold).faithfulness == 0.5


def test_facts_edge_cases():
    assert score_facts([], []).coverage == 0.0
    gold = [out("d", fact("writer", "age", "30 years old", "high"))]
    m = score_facts([], gold)  # no prediction for the document
    assert (m.coverage, m.faithfulness, m.high_coverage, m.support) == (0.0, 0.0, 0.0, 1)
    assert score_facts([out("d")], [out("d")]).coverage == 0.0  # no facts at all


# ---------------------------------------------------------------- rewrite


def decision(text, category, action, replacement=None):
    return RoutingDecision(
        span_text=text, category=category, action=action, replacement=replacement, reason="t"
    )


def rewritten(doc_id, safe, *decisions):
    return RewriterOutput(doc_id=doc_id, safe_text=safe, decisions=list(decisions), placeholders={})


def test_rewrite_clean_document():
    gold = [
        GoldRouting(
            doc_id="d",
            routing=[
                GoldRoute(text="Priya Raman", category=PIICategory.NAME, action=Action.PLACEHOLDER),
                GoldRoute(
                    text="vertigo", category=PIICategory.MEDICAL_CONDITION, action=Action.KEEP
                ),
            ],
        )
    ]
    facts = [out("d", fact("writer", "has_condition", "vertigo", "high"))]
    pred = [
        rewritten(
            "d",
            "[PERSON_1] has had vertigo.",
            decision("Priya Raman", PIICategory.NAME, Action.PLACEHOLDER, "[PERSON_1]"),
            decision("vertigo", PIICategory.MEDICAL_CONDITION, Action.KEEP),
        )
    ]
    m = score_rewrite(pred, gold, facts)
    assert (m.leak_rate, m.utility, m.routing_accuracy, m.support) == (0.0, 1.0, 1.0, 2)


def test_rewrite_every_span_leaked_and_variants():
    gold = [
        GoldRouting(
            doc_id="d",
            routing=[
                GoldRoute(text="Priya Raman", category=PIICategory.NAME, action=Action.PLACEHOLDER),
                GoldRoute(
                    text="+91 98450 12345", category=PIICategory.PHONE, action=Action.PLACEHOLDER
                ),
            ],
        )
    ]
    pred = [rewritten("d", "priya  raman called from +91-98450-12345")]
    m = score_rewrite(pred, gold, [])
    assert m.leak_rate == 1.0 and m.routing_accuracy == 0.0


def test_rewrite_utility_counts_generalised_form_and_loss():
    gold = [GoldRouting(doc_id="d", routing=[])]
    facts = [
        out(
            "d",
            fact("writer", "has_condition", "amyloidosis", "high"),
            fact("writer", "has_symptom", "swelling of the tongue", "high"),
            fact("writer", "age", "63 years old", "medium"),
        )
    ]
    pred = [
        rewritten(
            "d",
            "He has a rare protein disorder.",
            decision(
                "amyloidosis",
                PIICategory.MEDICAL_CONDITION,
                Action.GENERALISE,
                "a rare protein disorder",
            ),
        )
    ]
    m = score_rewrite(pred, gold, facts)
    assert m.utility == 0.5  # generalised condition kept, symptom lost; medium fact ignored


def test_rewrite_missing_prediction_and_quasi_combo_and_empty():
    gold = [
        GoldRouting(
            doc_id="d",
            routing=[
                GoldRoute(text="a + b", category=PIICategory.QUASI_COMBO, action=Action.GENERALISE),
                GoldRoute(text="Pune", category=PIICategory.LOCATION, action=Action.GENERALISE),
            ],
        )
    ]
    m = score_rewrite([], gold, [])
    assert m.leak_rate == 0.0 and m.routing_accuracy == 0.0 and m.support == 2
    pred = [
        rewritten(
            "d", "He lives in Pune.", decision("Pune", PIICategory.LOCATION, Action.GENERALISE)
        )
    ]
    assert score_rewrite(pred, gold, []).leak_rate == 1.0  # the combo route is skipped, Pune leaks
    assert score_rewrite([], [], []).support == 0


def test_rewrite_repeated_text_matched_in_order():
    gold = [
        GoldRouting(
            doc_id="d",
            routing=[
                GoldRoute(text="Houston", category=PIICategory.LOCATION, action=Action.GENERALISE),
                GoldRoute(text="Houston", category=PIICategory.LOCATION, action=Action.KEEP),
            ],
        )
    ]
    pred = [
        rewritten(
            "d",
            "x",
            decision("Houston", PIICategory.LOCATION, Action.GENERALISE),
            decision("Houston", PIICategory.LOCATION, Action.KEEP),
        )
    ]
    assert score_rewrite(pred, gold, []).routing_accuracy == 1.0


# --------------------------------------------------------------------- qa


def test_qa_accuracy_and_generalised_form():
    qa = [
        QAItem(
            qa_id="q1",
            question="What did the doctor suggest?",
            expected=["contact dermatitis"],
            doc_ids=["x"],
        ),
        QAItem(qa_id="q2", question="Who?", expected=["Priya"], doc_ids=["x"]),
    ]
    answers = [
        answer("It may be contact dermatitis.", "baseline", "What did the doctor suggest?"),
        answer("A medical condition.", "protected", "What did the doctor suggest?"),
        answer("Priya", "baseline", "Who?"),
        answer("[PERSON_1]", "protected", "Who?"),
        answer("ignored", "baseline", "not in the set"),
    ]
    m = score_qa(answers, qa)
    assert m.baseline_accuracy == 1.0 and m.protected_accuracy == 0.0 and m.support == 2
    assert score_qa([], []).baseline_accuracy == 0.0


# ----------------------------------------------------------------- attacks

PEOPLE = [
    Person(
        person_id="p1",
        name="Priya Raman",
        sex="F",
        age=34,
        age_band="30-39",
        town="Coonoor",
        district="Nilgiris",
        state="Tamil Nadu",
        occupation="pediatric neurosurgeon",
        occupation_group="doctor",
        conditions=["vertigo"],
        phone="+91 98450 12345",
    ),
    Person(
        person_id="p2",
        name="Vikram Das",
        sex="M",
        age=34,
        age_band="30-39",
        town="Shimla",
        district="Shimla",
        state="Himachal Pradesh",
        occupation="bank clerk",
        occupation_group="finance worker",
        conditions=["vertigo"],
        phone="+91 98160 30472",
    ),
    Person(
        person_id="p3",
        name="Anil Menon",
        sex="M",
        age=47,
        age_band="40-49",
        town="Coonoor",
        district="Nilgiris",
        state="Tamil Nadu",
        occupation="general physician",
        occupation_group="doctor",
        conditions=["hypertension"],
        phone=None,
    ),
    Person(
        person_id="p4",
        name="Pune Rao",
        sex="M",
        age=50,
        age_band="50-59",
        town="Pune",
        district="Pune",
        state="Maharashtra",
        occupation="farmer",
        occupation_group="agricultural worker",
        conditions=[],
        phone=None,
    ),
]


def test_round_pool_shrinks_and_identifies():
    r0 = score_round(PEOPLE, "p1", [answer("Nothing to say.")])
    assert (r0.disclosed, r0.candidate_pool, r0.identified, r0.recovery_rate) == ([], 4, False, 0.0)
    r1 = score_round(PEOPLE, "p1", [answer("A patient with vertigo was seen.")])
    assert r1.disclosed == ["condition_0"] and r1.candidate_pool == 2 and not r1.identified
    r2 = score_round(
        PEOPLE, "p1", [answer("A patient with vertigo was seen."), answer("She is 34 years old.")]
    )
    assert r2.candidate_pool == 2 and not r2.identified  # age 34 + vertigo still fits two people
    r3 = score_round(
        PEOPLE, "p1", [answer("34-year-old with vertigo works as a pediatric neurosurgeon")]
    )
    assert r3.candidate_pool == 1 and r3.identified and r3.recovery_rate == 0.5
    r4 = score_round(PEOPLE, "p1", [answer("call +91-98450-12345")])
    assert "phone" in r4.disclosed and r4.identified


def test_round_generalised_forms_and_false_hits():
    r = score_round(
        PEOPLE,
        "p1",
        [answer("A woman... a doctor in their 30s from a town in Tamil Nadu, Nilgiris district")],
    )
    assert {"age_band", "state", "district"} <= set(r.disclosed)
    assert "occupation_group" not in r.disclosed  # bare "doctor" without "works as a" is not a hit
    assert r.candidate_pool == 1 and r.recovery_rate == 0.0
    r2 = score_round(PEOPLE, "p1", [answer("She works as a doctor.")])
    assert "occupation_group" in r2.disclosed
    assert score_round(PEOPLE, "p4", [answer("Puneet lives in Pune?")]).disclosed.count("town") == 1
    assert "town" not in score_round(PEOPLE, "p4", [answer("Puneet lives nearby")]).disclosed
    assert (
        "age"
        not in score_round(PEOPLE, "p1", [answer("call 1234 5678 905 or 2034 years")]).disclosed
    )


def test_round_unknown_target_and_no_answers():
    r = score_round(PEOPLE, "nope", [answer("Priya Raman")])
    assert r.candidate_pool == 4 and r.identified is False
    assert score_round(PEOPLE, "p1", []).candidate_pool == 4


def test_target_info_counts_distinct_values():
    assert target_info(PEOPLE, []) == 0
    n = target_info(
        PEOPLE, [answer("Priya Raman lives in Coonoor"), answer("Coonoor again, Priya Raman")]
    )
    assert n == 2  # name and town; Coonoor is shared by two people and counts once
    assert target_info(PEOPLE, [answer("[PERSON_1] lives in a town in Tamil Nadu")]) == 0
