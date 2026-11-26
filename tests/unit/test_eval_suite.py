"""The evaluation suite's maths and question generators (tools/eval_suite)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from eval_suite import metrics as M
from eval_suite.e2e import value_found
from eval_suite.realdata import contract_name


def test_longest_common_run_counts_consecutive_tokens() -> None:
    a = "the patient Priya Raman lives in Coonoor and has vertigo since March"
    assert M.longest_common_run(a, "Priya Raman lives in Coonoor and has vertigo") == 8
    assert M.longest_common_run("", a) == 0


def test_rouge_l_and_token_f1() -> None:
    assert M.rouge_l("a b c d", "a b c d") == pytest.approx(1.0)
    assert M.rouge_l("a x b y", "a b") == pytest.approx(2 * 1.0 * 0.5 / 1.5)
    assert M.token_f1("Income was 41,721 pounds", "41721") == pytest.approx(2 * 0.25 / 1.25)
    assert M.token_f1("nothing", "else") == 0.0


def test_auc_and_tpr_at_fpr() -> None:
    assert M.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert M.auc([0.5], [0.5]) == 0.5
    assert M.tpr_at_fpr([0.9, 0.8, 0.3], [0.1, 0.2, 0.85], 0.01) == pytest.approx(1 / 3)


def test_statistics() -> None:
    assert M.mcnemar_p([(True, False)] * 10) < 0.01
    assert M.mcnemar_p([(True, True)] * 10) == 1.0
    assert M.wilcoxon_p([(1.0, 0.0)] * 20) < 0.001
    lo, hi = M.bootstrap_ci([0.0, 1.0] * 50, M.mean)
    assert lo < 0.5 < hi


def test_value_matching_for_end_to_end_labels() -> None:
    assert value_found("+91 98450 12345", "call 9845012345 today")
    assert value_found("Priya Raman", "Patient: PRIYA RAMAN, 34")
    assert not value_found("Pune", "Punekar Road")


def test_contract_names() -> None:
    assert contract_name("LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT") == (
        "distributor agreement of Limeenergyco"
    )
    assert contract_name("Magenta Therapeutics, Inc. - Master Agreement") == (
        "master agreement of Magenta Therapeutics, Inc."
    )


def test_attack_sessions_cover_three_protocols() -> None:
    from eval_suite import questions as Q
    from eval_suite.domains import get_domain

    repo = Path(__file__).resolve().parents[2]
    d = get_domain("health", repo)
    sessions = Q.attack_sessions(d, d.people(), 5)
    assert len(sessions) == 15
    assert {s["protocol"] for s in sessions} == {"direct", "progressive", "mixed"}
    assert all(len(s["queries"]) == Q.SESSION_ROUNDS for s in sessions)
    people = {p.person_id: p for p in d.people()}
    for s in sessions:  # the attacker's starting knowledge never names the target
        p = people[s["target"]]
        assert p.name not in " ".join(s["queries"])


def test_answer_score_finds_a_short_reference_in_a_full_answer() -> None:
    answer = "The consumer claims that a car loan was opened without their knowledge."
    assert M.answer_score(answer, "car loan") == 1.0
    assert M.answer_score("The records do not say.", "car loan") == 0.0
    long_ref = "consult a good dentist and go for a root canal treatment"
    assert M.answer_score("see a dentist", long_ref) == M.token_f1("see a dentist", long_ref)


def test_loose_found_reads_parts_and_bare_numbers_but_not_decades() -> None:
    from eval_suite.residual import loose_found

    assert loose_found("name", "Samuel Joseph", "please call samuel")
    assert loose_found("age", "66", "the patient, 66, reported")
    assert not loose_found("age", "60", "an adult in their 60s")
    assert loose_found("phone", "+91 94470 74015", "reach them on 470 74015")
    assert not loose_found("name", "Samuel Joseph", "[PERSON_1] reported numbness")


def test_person_membership_scores_values_given_back() -> None:
    from types import SimpleNamespace

    from eval_suite import report

    people = [SimpleNamespace(person_id="a", v="Asha"), SimpleNamespace(person_id="b", v="Bina")]
    d = SimpleNamespace(
        people=lambda: people,
        disclosed_exact=lambda p, text: {"name"} if p.v.lower() in text.lower() else set(),
    )
    rows = [
        {"person_id": "a", "group": "member", "query": "Who is A?",
         "baseline": {"answer": "Asha, 40"}, "protected": {"answer": "The records do not say."}},
        {"person_id": "b", "group": "nonmember", "query": "Who is B?",
         "baseline": {"answer": "The records do not say."},
         "protected": {"answer": "The records do not say."}},
    ]  # fmt: skip
    out = report.person_membership(d, rows)
    assert out["baseline"]["auc"] == 1.0 and out["protected"]["auc"] == 0.5


def test_scan_utility_questions_never_ask_for_what_privacy_removes() -> None:
    from eval_suite.e2e import _names_a_party, _private_kind

    for answer in ("October 15, 2004", "30th day of June 1999", "healthcentral.com", "952"):
        assert _private_kind(answer)
    for answer in ("30 days", "No acute cardiopulmonary process", "Rs. 21661.82"):
        assert not _private_kind(answer)
    assert _names_a_party("What state corporation is UNION DENTAL CORP. in the agreement?")
    assert not _names_a_party("What did the CT SCAN show?")


def test_question_writer_moves_to_the_next_model_when_every_key_is_limited(monkeypatch) -> None:
    import io
    import urllib.error

    from eval_suite import e2e

    asked: list[tuple[str, str]] = []

    def fake(prompt: str, model: str, key: str) -> str:
        asked.append((model, key))
        if model == "m1":
            raise urllib.error.HTTPError("u", 429, "Too Many Requests", {}, io.BytesIO(b"per day"))
        return '{"ok": true}'

    monkeypatch.setattr(e2e, "_gemini", fake)
    monkeypatch.setattr(e2e.time, "sleep", lambda s: None)
    monkeypatch.setenv("GEMINI_API_KEYS", "k1,k2")
    monkeypatch.setenv("EVAL_LABEL_MODEL", "m1")
    monkeypatch.setenv("SANITISE_AGENTS_MODEL", "m2")
    g = e2e.Gemini()
    assert g.ask("p", "q1") == {"ok": True}
    assert asked[:3] == [("m1", "k1"), ("m1", "k2"), ("m2", "k1")]  # not m1 again and again
