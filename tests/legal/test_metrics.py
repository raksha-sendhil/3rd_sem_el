"""The seven core metrics (Person D) and the legal attack scorer."""

from __future__ import annotations

from pathlib import Path

import pytest

from sanitise.attacks.runner import load_people
from sanitise.evaluation import legal as m
from sanitise.evaluation.attacks import score_round, target_info
from sanitise.legal.report import render
from sanitise.schemas import PIICategory, QAItem, RagAnswer, SensitiveSpan

CLIENTS = Path(__file__).resolve().parents[2] / "data" / "legal" / "clients.json"


def ans(text: str, pipeline: str = "baseline", query: str = "q", ms: int = 100) -> RagAnswer:
    return RagAnswer(
        pipeline=pipeline, query=query, answer=text, doc_ids=[], payload="", latency_ms=ms
    )


def test_rouge_l() -> None:
    assert m.rouge_l("the cat sat", "the cat sat") == pytest.approx(1.0)
    assert m.rouge_l("", "anything") == 0.0
    # LCS "the sat" = 2 of 3 tokens each side
    assert m.rouge_l("the cat sat", "the dog sat") == pytest.approx(2 / 3)


def test_extraction_leakage() -> None:
    doc = "Client Joseph Bose, phone 98450 12345, filed CC/118/2023 for a cheque bounce."
    rows = [
        (doc, doc, ["Joseph Bose", "+91 98450 12345", "CC/118/2023"]),
        ("[CLIENT_1] filed a complaint.", doc, ["Joseph Bose", "+91 98450 12345", "CC/118/2023"]),
    ]
    out = m.extraction_leakage(rows)
    assert out["exact_identifier_matches"] == 2  # name and case number; phone digits differ
    assert out["leaked_chunks"] == pytest.approx(0.5)


def test_pii_leakage_ignores_placeholders() -> None:
    def detect(text: str) -> list[SensitiveSpan]:
        i = text.find("Bose")
        if i < 0:
            return []
        return [
            SensitiveSpan(text="Bose", category=PIICategory.NAME, start=i, end=i + 4, method="rule")
        ]

    out = m.pii_leakage(["Mr. Bose agreed.", "[CLIENT_1] agreed.", "No names."], detect)
    assert out["rate"] == pytest.approx(1 / 3) and out["by_category"] == {"NAME": 1}


def test_membership_inference_auc_and_tpr() -> None:
    perfect = m.membership_inference([0.9, 0.8], [0.1, 0.2])
    assert perfect["auc"] == 1.0 and perfect["tpr_at_1pct_fpr"] == 1.0
    chance = m.membership_inference([0.5, 0.5], [0.5, 0.5])
    assert chance["auc"] == 0.5 and chance["tpr_at_1pct_fpr"] == 0.0


def test_recall_at_k_and_answer_quality() -> None:
    qa = [
        QAItem(qa_id="q1", question="a", expected=["notice"], doc_ids=["d1"]),
        QAItem(qa_id="q2", question="b", expected=["notice"], doc_ids=["d9"]),
    ]
    rk = m.recall_at_k({"a": ["d1", "d2"], "b": ["d3", "d4", "d5", "d9"]}, qa, ks=(1, 4))
    assert rk == {1: 0.5, 4: 1.0}
    quality = m.answer_quality(
        [ans("Send a statutory demand-notice.", query="a"), ans("No.", "protected", "a")],
        [QAItem(qa_id="q1", question="a", expected=["statutory demand notice"])],
    )
    assert quality["baseline_accuracy"] == 1.0 and quality["retention"] == 0.0


def test_latency_ratio() -> None:
    out = m.latency([ans("x", ms=100), ans("x", "protected", ms=110)], {"agent1": [2000, 4000]})
    assert out["online_ratio"] == pytest.approx(1.1)
    assert out["offline_seconds_per_doc"] == {"agent1": 3.0}


# ---------------------------------------------------------------- attack scoring


def test_generalised_answers_narrow_the_pool_but_recover_no_secret() -> None:
    people = load_people(CLIENTS)
    coarse = score_round(
        people, "c001", [ans("A doctor in their 50s in a city in Maharashtra.", "protected")]
    )
    assert {"occupation_group", "age_band", "state"} <= set(coarse.disclosed)
    assert coarse.recovery_rate == 0.0
    assert coarse.identified  # doctor + 50s + Maharashtra is one client: why combos climb 2 steps
    vague = score_round(people, "c001", [ans("A professional, middle-aged, in western India.")])
    assert vague.candidate_pool > 1 and not vague.identified


def test_direct_identifiers_identify_however_they_are_written() -> None:
    people = load_people(CLIENTS)
    c001 = next(p for p in people if p.person_id == "c001")
    spaced = c001.case_number.replace(" ", "  ")
    for text in (c001.email, f"PAN {c001.pan.lower()}", f"case {spaced}"):
        assert score_round(people, "c001", [ans(text)]).identified, text
    assert target_info(people, [ans("[CLIENT_1] in a city in Maharashtra")]) == 0


# ---------------------------------------------------------------- report


def test_report_renders_every_section() -> None:
    pipe = {
        "extraction": {"mean_rouge_l": 0.1, "exact_identifier_matches": 0, "leaked_chunks": 0.0},
        "pii_leakage": {"rate": 0.0, "by_category": {}},
        "membership": {"auc": 0.55, "tpr_at_1pct_fpr": 0.02, "roc": [(0.0, 0.0), (1.0, 1.0)]},
    }
    results = {
        "pipelines": {"baseline": pipe, "protected": pipe},
        "answer_quality": {
            "baseline_accuracy": 0.9,
            "protected_accuracy": 0.85,
            "gap_points": 5.0,
            "retention": 0.94,
        },
        "latency": {"online_ratio": 1.05, "offline_seconds_per_doc": {}},
        "recall_at_k": {"1": 0.6, "2": 0.8, "4": 0.9, "8": 1.0},
        "agents": {
            "agent1": {"direct_recall": 0.99, "precision": 0.97},
            "agent2": {"high_coverage": 0.85, "faithfulness": 0.93},
            "agent3": {
                "leak_rate": 0.01,
                "routing_accuracy": 0.9,
                "utility": 0.88,
                "documents_with_direct_leaks": 0,
            },
        },
        "attack_outcomes": [
            {
                "attack_id": "stepwise_c001",
                "monitor": "monitor",
                "baseline_identified_at": 3,
                "protected_identified_at": None,
                "protected_min_pool": 12,
                "blocked_rounds": [2, 3, 4],
            }
        ],
    }
    text = render(results)
    for needle in ("Core metrics", "Agent-level", "Attack outcomes", "| stepwise_c001 |", "never"):
        assert needle in text
    assert "**no**" not in text
