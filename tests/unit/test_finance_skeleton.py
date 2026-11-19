"""The finance skeleton: its package, the four sources' sample data and the data step."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from eval_suite import questions as Q  # noqa: E402
from eval_suite.domains import FINANCE_SOURCES, get_domain  # noqa: E402

SCALE = {"indexed": 12, "heldout": 6, "targeted": 5, "repeat": 5, "members": 5, "nonmembers": 5,
         "utility": 5, "targets": 1, "e2e": 2, "gen_docs": 15}  # fmt: skip


def test_the_finance_package_is_its_own() -> None:
    from sanitise_finance.agents.rewriter.ontology import ONTOLOGY_DIR
    from sanitise_finance.config import Settings
    from sanitise_finance.schemas import Person

    assert ONTOLOGY_DIR == ROOT / "data" / "finance" / "ontology"
    assert Settings().store_path == "data/finance/store.sqlite"
    # a bank population may lack age and job
    p = Person(person_id="b1", name="Asha Rao", bank="Canara Bank", account_number="5010023")
    assert p.age is None and p.bank == "Canara Bank"


def test_account_numbers_are_direct_identifiers() -> None:
    from sanitise_finance.evaluation.attacks import DIRECT, _found

    assert "account_number" in DIRECT
    assert _found("account_number", "50100234567", "a/c 5010 0234 567")


@pytest.mark.parametrize("source", FINANCE_SOURCES)
def test_each_source_reads_its_folder(source: str, tmp_path: Path) -> None:
    from eval_suite import finance

    d = get_domain("finance", ROOT, source)
    finance.step_data(d, ROOT, tmp_path, SCALE, Q)
    data = tmp_path / "data"
    sessions = json.loads((data / "sessions.json").read_text(encoding="utf-8"))
    if source == "bank":  # the population: attack sessions, Target Info on people
        assert sessions and all(len(s["queries"]) == Q.SESSION_ROUNDS for s in sessions)
        assert list((data / "gen" / "corpus").glob("*.json"))
    else:
        assert sessions == []
    if source != "cheques":  # text sources: a store and its questions
        assert list((data / "real" / "corpus").glob("*.json"))
    if source in ("kyc", "corporate"):  # Target Info on the dataset's own labels
        targeted = json.loads((data / "targeted.json").read_text(encoding="utf-8"))
        assert targeted and all(t["store"] == "real" and t["values"] for t in targeted)


def test_finance_sessions_narrow_by_bank_and_branch() -> None:
    d = get_domain("finance", ROOT, "bank")
    sessions = Q.attack_sessions(d, d.people(), 1)
    progressive = next(s for s in sessions if s["protocol"] == "progressive")["queries"]
    assert any("bank" in q.lower() for q in progressive)
    assert any("account number" in q for q in progressive)
