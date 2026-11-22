"""The finance pipeline (Person B): identities go, amounts, products and issues stay.

Every text here is made up for these tests.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import pytest

from sanitise_finance.agents.extractor import importance
from sanitise_finance.agents.extractor.pipeline import check_facts
from sanitise_finance.agents.rewriter import ontology, prompt, routing
from sanitise_finance.agents.rewriter.pipeline import rewrite_with_trace
from sanitise_finance.rules import rule_detect
from sanitise_finance.rules.finance import luhn
from sanitise_finance.schemas import (
    Action,
    DetectorOutput,
    ExtractorOutput,
    FactTuple,
    Importance,
    Subject,
)
from sanitise_finance.schemas import PIICategory as C

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))


def removed(text: str) -> list[str]:
    decisions = routing.route(rule_detect(text), [])
    return [d.span_text for d in decisions if d.action is not Action.KEEP]


def fact(relation: str, value: str, subject: str = "writer") -> FactTuple:
    return FactTuple(subject=subject, relation=relation, value=value,
                     importance=importance.default_importance(relation))  # fmt: skip


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("My account number is 50100998877 at the branch.", "50100998877"),
        ("A/c No: 004512340000", "004512340000"),
        ("The card 4111 1111 1111 1111 was charged.", "4111 1111 1111 1111"),
        ("IFSC code is ABCD0123456.", "ABCD0123456"),
        ("Her PAN is ABCPK4321Q.", "ABCPK4321Q"),
        ("Pay me at kiran.p@okaxis today.", "kiran.p@okaxis"),
        ("IBAN: GB29NWBK60161331926819", "GB29NWBK60161331926819"),
        ("MICR code: 560002999", "560002999"),
        ("Customer ID: CU-778812", "CU-778812"),
        ("Loan number: HL-2093318", "HL-2093318"),
        ("SSN 219-09-9999 on file", "219-09-9999"),
        ("SWIFT code: ABCDGB2L123", "ABCDGB2L123"),
        ("Account PIN: 4821", "4821"),
        ("Address: 41 SEFTON ROAD", "41 SEFTON ROAD"),
    ],
)
def test_identifiers_are_removed(text: str, value: str) -> None:
    assert value in " | ".join(removed(text))


def test_long_digit_runs_do_not_hang_the_date_filter() -> None:
    import time

    from sanitise_finance.rules.engine import _VAGUE_DATE

    start = time.perf_counter()
    for text in ("2" * 400 + "Z", "1-" * 200 + "Z", "12 " * 200 + "Z"):
        assert not _VAGUE_DATE.match(text)
    for vague in ("for 2 days", "2-3 weeks", "1.5 hours", "the last 12 months"):
        assert _VAGUE_DATE.match(vague)
    assert time.perf_counter() - start < 5


def test_luhn_rejects_a_wrong_checksum() -> None:
    assert luhn("4111 1111 1111 1111")
    assert not luhn("4111 1111 1111 1112")


def test_amounts_products_terms_and_report_years_survive() -> None:
    text = ("Home loan of 25,00,000 over 20 years at 8.5%. Late fee ₹18,400.\n"
            "Total finance costs   5,205   4,705\nReported for 2019 and 2018, in Q3.")  # fmt: skip
    gone = " | ".join(removed(text))
    for kept in ("Home loan", "25,00,000", "20 years", "8.5", "18,400", "5,205", "4,705", "2019", "Q3"):
        assert kept not in gone


def test_a_date_of_birth_is_still_generalised() -> None:
    assert any("1988" in s for s in removed("Date of birth: 14 March 1988"))


def test_a_name_does_not_run_into_the_next_label() -> None:
    spans = rule_detect("Signature: John Alexander Doe\nDate: 01/10/2023")
    assert [s.text for s in spans if s.category is C.NAME] == ["John Alexander Doe"]


def test_a_detail_is_hidden_everywhere_it_is_written() -> None:
    text = "Lender: Corvane Partners\nThe borrower repays Corvane Partners monthly."
    assert removed(text).count("Corvane Partners") == 2


def test_ladders() -> None:
    def steps(text: str, category: C) -> list[str]:
        return [ontology.generalise(text, category, level=n) for n in (1, 2)]

    assert steps("₹18,400", C.AMOUNT) == ["about ₹20,000", "a five-figure sum"]
    assert steps("Canara Bank", C.LOCATION) == ["a public-sector bank", "a bank"]
    assert steps("home loan", C.PRODUCT) == ["a secured loan", "a financial product"]


def test_useful_relations() -> None:
    for relation in ("product", "issue", "amount", "account_event", "income"):
        assert importance.default_importance(relation) is Importance.HIGH


def test_agent2_drops_invented_figures_and_undetected_employers() -> None:
    text = "The fee was 1,25,000 and the limit is 900. I work at Zed."
    facts = [fact("amount", "fee was 125000"), fact("amount", "limit is 950"), fact("employer", "Zed")]
    kept = check_facts(text, facts, [], [Subject(id="writer", label="the writer")], trust_subjects=True)
    assert [f.value for f in kept] == ["fee was 125000"]


def test_a_short_removed_number_does_not_hit_inside_an_amount() -> None:
    spans = rule_detect("Account PIN: 5205")
    decisions = routing.route(spans, [])
    for d in decisions:
        d.replacement = d.replacement or "[ID_1]"
    assert prompt.safe_value("total: 15,5205.50 and 5205", decisions) == "total: 15,5205.50 and [ID_1]"


class _Forgetful:
    """A rewrite model that drops every figure."""

    model = "fake"

    async def text_call(self, system: str, user: str) -> str:
        return "The record is about a loan."


def test_a_figure_the_rewrite_drops_is_added_back(monkeypatch: pytest.MonkeyPatch) -> None:
    from sanitise_finance.agents.rewriter import pipeline

    monkeypatch.setattr(pipeline, "agents_llm", lambda llm: llm)
    trace = asyncio.run(
        rewrite_with_trace(
            "d",
            DetectorOutput(doc_id="d", spans=[]),
            ExtractorOutput(doc_id="d", facts=[fact("amount", "loan amount: 12,500")]),
            _Forgetful(),
        )
    )
    assert "12,500" in trace.output.safe_text


def test_answer_prompt_is_about_financial_records() -> None:
    from sanitise_finance.rag.answer_prompt import SYSTEM_PROMPT

    assert "financial" in SYSTEM_PROMPT and "patient" not in SYSTEM_PROMPT


def test_indian_grouping_is_one_token() -> None:
    from eval_suite.metrics import tokens

    assert tokens("1,25,000") == ["125000"] == tokens("125000") == tokens("1,25000/-")
    assert tokens("41,721") == ["41721"]
    assert tokens("4, 5 and 6") == ["4", "5", "and", "6"]


def test_scans_of_one_type_get_different_questions() -> None:
    from eval_suite.finance import e2e_questions

    same = "What was the total income?"
    rows = [
        {"id": str(i), "type": "annual_report", "private": ["chair"],
         "fields": {"chair": f"Person {i}", "income": f"{i}1,000", "spending": f"{i}9,500"},
         "questions": [{"question": same, "answer": f"{i}1,000"}]}
        for i in (1, 2)
    ]  # fmt: skip
    out = e2e_questions(rows, "corporate")
    first, second = out["1"][1][0]["question"], out["2"][1][0]["question"]
    assert first != second and "19,500" in first and "11,000" not in first
    assert out["1"][0][0]["question"] != out["2"][0][0]["question"]


def test_kyc_data_meets_the_plan() -> None:
    base = ROOT / "data" / "finance" / "kyc"
    docs = [json.loads(f.read_text(encoding="utf-8")) for f in (base / "docs").glob("*.json")]
    assert len(docs) == 100 and len(list((base / "heldout").glob("*.json"))) == 60
    assert all(len(d["text"]) <= 1900 and d["private"] and d["meta"]["synthetic"] for d in docs)
    texts = {d["doc_id"]: " ".join(d["text"].split()) for d in docs}
    utility = json.loads((base / "utility.json").read_text(encoding="utf-8"))
    targeted = json.loads((base / "targeted.json").read_text(encoding="utf-8"))
    assert len(utility) >= 60 and len(targeted) >= 60
    assert all(u["reference"] in texts[u["doc_id"]] for u in utility)
    private = [v.lower() for d in docs for v in d["private"] if len(v) >= 4]
    for q in utility + targeted:
        low = q["query"].lower()
        named = [v for v in private if v in low and re.search(rf"(?<!\w){re.escape(v)}(?!\w)", low)]
        assert not named, q["query"]
    gold = (base / "gold.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(gold) >= 20 and all((base / json.loads(g)["file"]).is_file() for g in gold)
