"""Contract sections: role words and durations are content, clause amounts are kept."""

from sanitise_legal.agents.rewriter.prompt import restore, safe_value
from sanitise_legal.agents.rewriter.routing import route
from sanitise_legal.rules.content import not_identifying
from sanitise_legal.schemas import Action, FactTuple, Importance, PIICategory, SensitiveSpan

C = PIICategory
TEXT = "The Supplier, Acme Fibre Ltd, gives thirty days notice from 4 May 2021; cap $50,000."


def span(words, category):
    start = TEXT.index(words)
    return SensitiveSpan(
        text=words, category=category, start=start, end=start + len(words), method="rule"
    )


def test_role_words_and_durations_are_not_identifying():
    assert not_identifying(TEXT, span("Supplier", C.ORGANIZATION))
    assert not_identifying(TEXT, span("thirty days", C.DATE))
    assert not not_identifying(TEXT, span("Acme Fibre Ltd", C.ORGANIZATION))
    assert not not_identifying(TEXT, span("4 May 2021", C.DATE))


def test_a_clause_amount_is_kept_and_a_claim_amount_is_not():
    money = span("$50,000", C.MONEY)
    fact = FactTuple(
        subject="client", relation="clause", value="cap $50,000", importance=Importance.HIGH
    )
    assert route([money], [fact])[0].action is Action.KEEP
    assert route([money], [])[0].action is Action.GENERALISE
    assert "The contract provides: cap $50,000." in restore("A supply deal.", [fact], [])


def test_replacement_is_by_whole_words():
    d = route([span("Acme Fibre Ltd", C.ORGANIZATION)], [])[0]
    d.span_text, d.replacement = "US", "a place"
    assert safe_value("the business in US", [d]) == "the business in a place"
