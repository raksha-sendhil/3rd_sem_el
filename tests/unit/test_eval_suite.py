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
