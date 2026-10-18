import itertools

import pytest

from sanitise.rules.categories import CATEGORY_PRECEDENCE, precedence
from sanitise.rules.normalise import OffsetOutOfRangeError, normalise_spans, trim
from sanitise.schemas import PIICategory as C
from sanitise.schemas import SensitiveSpan


def at(doc: str, needle: str, cat: C, conf: float = 1.0, occurrence: int = 0) -> SensitiveSpan:
    start = -1
    for _ in range(occurrence + 1):
        start = doc.index(needle, start + 1)
    return SensitiveSpan(
        text=needle,
        category=cat,
        start=start,
        end=start + len(needle),
        method="rule",
        confidence=conf,
    )


def triples(spans):
    return [(s.text, s.category, s.start) for s in spans]


def test_precedence_covers_every_category_once():
    assert sorted(CATEGORY_PRECEDENCE, key=lambda c: c.value) == sorted(C, key=lambda c: c.value)
    assert precedence(C.AGE) < precedence(C.DATE)
    assert precedence(C.ADDRESS) < precedence(C.LOCATION)
    assert precedence(C.EMAIL) < precedence(C.OTHER)


def test_offsets_out_of_range_raise():
    doc = "short"
    bad = SensitiveSpan(text="short!", category=C.NAME, start=0, end=6, method="rule")
    with pytest.raises(OffsetOutOfRangeError):
        normalise_spans([bad], doc)


def test_offsets_text_mismatch_raise():
    doc = "Alice met Bob"
    bad = SensitiveSpan(text="Bob", category=C.NAME, start=0, end=3, method="rule")
    with pytest.raises(OffsetOutOfRangeError):
        normalise_spans([bad], doc)


def test_trim_strips_punctuation_and_possessive():
    doc = "Ask Sally Beck's office."
    s = at(doc, "Sally Beck's ", C.NAME)
    t = trim(s)
    assert (t.text, t.start, t.end) == ("Sally Beck", 4, 14)
    assert doc[t.start : t.end] == t.text


def test_trim_drops_pure_punctuation():
    doc = "x , y"
    assert trim(at(doc, " , ", C.OTHER)) is None


def test_exact_duplicates_collapse_keeping_best_confidence():
    doc = "Call 555-123-4567 now"
    a = at(doc, "555-123-4567", C.PHONE, conf=0.4).model_copy(
        update={"rationale": "PhoneRecognizer"}
    )
    b = at(doc, "555-123-4567", C.PHONE, conf=0.75).model_copy(update={"rationale": "regex"})
    [out] = normalise_spans([a, b], doc)
    assert out.confidence == 0.75
    assert out.rationale == "PhoneRecognizer; regex"


def test_same_category_overlap_unions_range():
    doc = "Dr Robert James Hale arrived"
    a = at(doc, "Robert James", C.NAME)
    b = at(doc, "James Hale", C.NAME)
    [out] = normalise_spans([a, b], doc)
    assert out.text == "Robert James Hale"
    assert doc[out.start : out.end] == out.text


def test_precedence_age_beats_date():
    doc = "He is 32 years old."
    out = normalise_spans([at(doc, "32 years old", C.DATE), at(doc, "32 years old", C.AGE)], doc)
    assert triples(out) == [("32 years old", C.AGE, 6)]


def test_precedence_beats_length_when_contained():
    doc = "jane.doe@gmail.com"
    out = normalise_spans(
        [at(doc, "gmail.com", C.OTHER), at(doc, "jane.doe@gmail.com", C.EMAIL)], doc
    )
    assert triples(out) == [("jane.doe@gmail.com", C.EMAIL, 0)]


def test_address_absorbs_location():
    doc = "We live at 42 Elm Street, Springfield, IL 62704."
    out = normalise_spans(
        [
            at(doc, "Springfield", C.LOCATION),
            at(doc, "42 Elm Street, Springfield, IL 62704", C.ADDRESS),
        ],
        doc,
    )
    assert [s.category for s in out] == [C.ADDRESS]


def test_partial_overlap_keeps_meaningful_residual():
    # A weaker span that sticks out past a stronger one keeps the unclaimed part.
    doc = "seen March 3, 2021 Springfield clinic"
    date = at(doc, "March 3, 2021", C.DATE)
    loc = at(doc, "2021 Springfield", C.LOCATION)
    out = normalise_spans([loc, date], doc)
    assert triples(out) == [("March 3, 2021", C.DATE, 5), ("Springfield", C.LOCATION, 19)]
    for s in out:
        assert doc[s.start : s.end] == s.text


def test_partial_overlap_drops_stopword_residual():
    doc = "my 5 year old son"
    rel = at(doc, "my 5 year old", C.RELATION)
    age = at(doc, "5 year old", C.AGE)
    out = normalise_spans([rel, age], doc)
    assert triples(out) == [("5 year old", C.AGE, 3)]


def test_adjacent_same_category_merge():
    doc = "From John Smith today"
    out = normalise_spans([at(doc, "John", C.NAME), at(doc, "Smith", C.NAME)], doc)
    assert triples(out) == [("John Smith", C.NAME, 5)]


def test_adjacent_names_separated_by_comma_do_not_merge():
    doc = "cc John, Mary"
    out = normalise_spans([at(doc, "John", C.NAME), at(doc, "Mary", C.NAME)], doc)
    assert len(out) == 2


def test_adjacent_address_parts_merge_across_comma_and_full_stop():
    doc = "Send to 1400 Smith St., Houston, TX 77002 asap"
    out = normalise_spans(
        [at(doc, "1400 Smith St", C.ADDRESS), at(doc, "Houston, TX 77002", C.ADDRESS)], doc
    )
    assert triples(out) == [("1400 Smith St., Houston, TX 77002", C.ADDRESS, 8)]


def test_adjacent_different_categories_do_not_merge():
    doc = "Robert Hale Springfield"
    out = normalise_spans([at(doc, "Robert Hale", C.NAME), at(doc, "Springfield", C.LOCATION)], doc)
    assert len(out) == 2


def test_output_sorted_non_overlapping_and_idempotent():
    doc = "Call Jane Doe at 555-123-4567 on March 3, 2021 in Springfield"
    spans = [
        at(doc, "Springfield", C.LOCATION),
        at(doc, "March 3", C.DATE),
        at(doc, "3, 2021", C.DATE),
        at(doc, "555-123-4567", C.PHONE),
        at(doc, "123-4567", C.ID_NUMBER),
        at(doc, "Jane Doe", C.NAME),
    ]
    out = normalise_spans(spans, doc)
    starts = [s.start for s in out]
    assert starts == sorted(starts)
    for a, b in itertools.pairwise(out):
        assert a.end <= b.start
    assert normalise_spans(out, doc) == out
    assert ("123-4567", C.ID_NUMBER, doc.index("123-4567")) in triples(out)  # ID outranks PHONE


def test_non_positional_spans_pass_through():
    doc = "text"
    q = SensitiveSpan(
        text="rare job + rare condition", category=C.QUASI_COMBO, method="llm", rationale="r"
    )
    assert normalise_spans([q], doc) == [q]
