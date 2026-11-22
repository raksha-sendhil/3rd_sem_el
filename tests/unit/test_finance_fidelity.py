"""Finance: who said what in the third person, the detection gate, the product is content."""

from sanitise_finance.agents.extractor import speakers
from sanitise_finance.agents.rewriter import fidelity, prompt
from sanitise_finance.monitor.attributes import CONTENT_TYPES
from sanitise_finance.rules.content import not_identifying
from sanitise_finance.schemas import FactTuple, Importance, PIICategory, SensitiveSpan

TURNS = [
    "Customer: I was charged twice for my car loan instalment this May.",
    "Agent: We will reverse the second debit within five working days.",
]
TEXT = chr(10).join(TURNS)


def facts() -> list[FactTuple]:
    made = [
        FactTuple(subject="writer", relation=r, value=v, importance=Importance.HIGH)
        for r, v in [
            ("issue", "charged twice for my car loan instalment"),
            ("plan", "reverse the second debit within five working days"),
        ]
    ]
    return speakers.attribute(TEXT, made)


def test_the_draft_says_who_said_what_in_the_third_person() -> None:
    assert [f.said_by for f in facts()] == ["customer", "agent"]
    lines = prompt.draft(facts(), []).splitlines()
    assert lines == [
        "the customer - issue (said by the customer): "
        "charged twice for their car loan instalment.",
        "the customer - advice (said by the agent): "
        "reverse the second debit within five working days.",
    ]


def test_first_person_is_repaired_by_code() -> None:
    safe = prompt.safe_facts(facts(), [])
    fixed = fidelity.repair("I was charged twice.", safe, prompt.subject_labels(safe))
    assert "The customer reported: charged twice for their car loan" in fixed
    assert "The agent advised: reverse the second debit" in fixed


def test_gate_drops_roles_and_durations_only() -> None:
    text = "The agent said Lump Kumar gets a refund in five days, by 4 May 2021."

    def span(words: str, category: PIICategory) -> SensitiveSpan:
        start = text.index(words)
        return SensitiveSpan(
            text=words, category=category, start=start, end=start + len(words), method="rule"
        )

    assert not_identifying(text, span("agent", PIICategory.OCCUPATION))
    assert not_identifying(text, span("five days", PIICategory.DATE))
    assert not not_identifying(text, span("4 May 2021", PIICategory.DATE))
    assert not not_identifying(text, span("Lump", PIICategory.NAME))


def test_the_product_is_content_and_an_illness_is_personal() -> None:
    assert CONTENT_TYPES == {"product"}
