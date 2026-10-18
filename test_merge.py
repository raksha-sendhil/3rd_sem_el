from sanitise.rules.merge import merge_spans
from sanitise.schemas import PIICategory as C
from sanitise.schemas import SensitiveSpan

DOC = "My son is 32 years old, 6 ft 3 inches tall, and works as a lighthouse keeper in Kirkwall."


def rule(text: str, cat: C) -> SensitiveSpan:
    start = DOC.index(text)
    return SensitiveSpan(text=text, category=cat, start=start, end=start + len(text), method="rule")


def llm(text: str, cat: C = C.QUASI_COMBO) -> SensitiveSpan:
    return SensitiveSpan(text=text, category=cat, method="llm", confidence=0.7, rationale="because")


RULES = [rule("My son", C.RELATION), rule("32 years old", C.AGE), rule("6 ft 3 inches", C.PHYSICAL)]


def test_union_keeps_rule_spans_then_inferred_in_model_order():
    q1 = llm("lighthouse keeper + Kirkwall")
    q2 = llm("approximately 10 weeks of unresolved symptoms")
    out = merge_spans(RULES, [q1, q2])
    assert out == RULES + [q1, q2]


def test_drops_quasi_that_restates_one_explicit_identifier():
    out = merge_spans(RULES, [llm("32 years old"), llm("32 Years Old.", C.AGE)])
    assert out == RULES


def test_drops_combo_whose_every_part_is_explicit():
    out = merge_spans(RULES, [llm("32 years old + 6 ft 3 inches")])
    assert out == RULES


def test_keeps_combo_that_adds_new_information():
    q = llm("32 years old + lighthouse keeper")
    assert merge_spans(RULES, [q]) == RULES + [q]


def test_substring_of_explicit_counts_as_restatement():
    assert merge_spans(RULES, [llm("son", C.RELATION)]) == RULES


def test_partial_word_is_not_a_restatement():
    q = llm("2 years", C.DATE)  # "2 years" is not a word-bounded piece of "32 years old"
    assert merge_spans(RULES, [q]) == RULES + [q]


def test_deduplicates_inferred_by_category_and_normalised_text():
    a = llm("Lighthouse keeper in Kirkwall")
    b = llm("lighthouse keeper in kirkwall.")
    c = llm("lighthouse keeper in kirkwall", C.OCCUPATION)
    assert merge_spans(RULES, [a, b, c]) == RULES + [a, c]


def test_blank_llm_span_dropped():
    assert merge_spans(RULES, [llm("   ")]) == RULES


def test_positional_llm_span_goes_through_precedence():
    start = DOC.index("32 years")
    stray = SensitiveSpan(
        text="32 years", category=C.DATE, start=start, end=start + 8, method="llm", rationale="x"
    )
    assert merge_spans(RULES, [stray]) == RULES


def test_no_llm_spans_returns_rules_sorted():
    assert merge_spans(list(reversed(RULES)), []) == RULES


def test_empty_inputs():
    assert merge_spans([], []) == []
