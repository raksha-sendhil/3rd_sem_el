"""Unit tests for the scorer and the gold validator (Person D)."""

import json

import pytest

from sanitise.evaluation.scoring import match_document, score
from sanitise.evaluation.validate_gold import validate_file
from sanitise.schemas import DetectorOutput, PIICategory, SensitiveSpan


def sp(cat, start=None, end=None, text="x", method=None):
    if method is None:
        method = "rule" if start is not None else "llm"
    return SensitiveSpan(
        text=text,
        category=PIICategory(cat),
        start=start,
        end=end,
        method=method,
        rationale="r" if method == "llm" else None,
    )


def out(spans, doc_id="d1"):
    return DetectorOutput(doc_id=doc_id, spans=spans)


# ---------------------------------------------------------------- core behaviours
def test_perfect_match():
    g = [out([sp("AGE", 10, 22), sp("NAME", 0, 5)])]
    m = score(g, g)
    assert (m.precision, m.recall, m.f1, m.support) == (1.0, 1.0, 1.0, 2)


def test_partial_overlap_counts():
    gold = [out([sp("AGE", 10, 22)])]
    pred = [out([sp("AGE", 15, 30)])]  # overlaps 15-22
    m = score(pred, gold)
    assert m.recall == 1.0 and m.precision == 1.0


def test_adjacent_but_not_overlapping_does_not_match():
    gold = [out([sp("AGE", 10, 20)])]
    pred = [out([sp("AGE", 20, 30)])]  # touches at 20, zero overlap (end exclusive)
    m = score(pred, gold)
    assert m.recall == 0.0 and m.precision == 0.0


def test_wrong_category_does_not_match():
    gold = [out([sp("AGE", 10, 22)])]
    pred = [out([sp("DATE", 10, 22)])]
    m = score(pred, gold)
    assert m.recall == 0.0 and m.precision == 0.0
    assert m.per_category["AGE"]["fn"] == 1
    assert m.per_category["DATE"]["fp"] == 1


def test_extra_prediction_lowers_precision_only():
    gold = [out([sp("AGE", 10, 22)])]
    pred = [out([sp("AGE", 10, 22), sp("NAME", 30, 40)])]
    m = score(pred, gold)
    assert m.recall == 1.0
    assert m.precision == pytest.approx(0.5)


def test_missed_span_lowers_recall_only():
    gold = [out([sp("AGE", 10, 22), sp("NAME", 30, 40)])]
    pred = [out([sp("AGE", 10, 22)])]
    m = score(pred, gold)
    assert m.precision == 1.0
    assert m.recall == pytest.approx(0.5)


def test_empty_gold_and_empty_pred():
    m = score([out([])], [out([])])
    assert (m.precision, m.recall, m.f1, m.support) == (0.0, 0.0, 0.0, 0)


def test_empty_gold_with_prediction_is_false_positive():
    m = score([out([sp("NAME", 0, 4)])], [out([])])
    assert m.precision == 0.0 and m.support == 0
    assert m.per_category["NAME"]["fp"] == 1


# ---------------------------------------------------------------- one-to-one
def test_duplicate_predictions_do_not_double_count():
    gold = [out([sp("AGE", 10, 22)])]
    pred = [out([sp("AGE", 10, 22), sp("AGE", 11, 21), sp("AGE", 10, 22)])]
    m = score(pred, gold)
    assert m.recall == 1.0
    assert m.precision == pytest.approx(1 / 3)


def test_one_big_prediction_cannot_cover_two_gold_spans():
    gold = [out([sp("PHYSICAL", 10, 20), sp("PHYSICAL", 25, 35)])]
    pred = [out([sp("PHYSICAL", 10, 35)])]
    m = score(pred, gold)
    assert m.recall == pytest.approx(0.5)
    assert m.precision == 1.0


def test_best_overlap_is_preferred():
    gold = out([sp("NAME", 0, 10)])
    pred = out([sp("NAME", 8, 12), sp("NAME", 0, 10)])
    dm = match_document(pred, gold)
    assert len(dm.tp) == 1
    assert (dm.tp[0][0].start, dm.tp[0][0].end) == (0, 10)
    assert len(dm.fp) == 1


# ---------------------------------------------------------------- quasi-identifiers
def test_quasi_combo_matches_on_category_only():
    gold = [out([sp("QUASI_COMBO", text="rare job + condition")])]
    pred = [out([sp("QUASI_COMBO", text="totally different wording")])]
    m = score(pred, gold)
    assert m.recall == 1.0 and m.precision == 1.0


def test_quasi_count_is_one_to_one():
    gold = [out([sp("QUASI_COMBO", text="a")])]
    pred = [out([sp("QUASI_COMBO", text="a"), sp("QUASI_COMBO", text="b")])]
    m = score(pred, gold)
    assert m.recall == 1.0 and m.precision == pytest.approx(0.5)


def test_explicit_never_matches_offsetless():
    gold = [out([sp("QUASI_COMBO", text="combo")])]
    pred = [out([sp("QUASI_COMBO", 0, 5, text="combo", method="rule")])]
    m = score(pred, gold)
    assert m.recall == 0.0 and m.precision == 0.0


# ---------------------------------------------------------------- multi-document
def test_documents_are_scored_independently():
    # Same offsets in different documents must not cross-match.
    gold = [out([sp("AGE", 0, 5)], "a"), out([], "b")]
    pred = [out([], "a"), out([sp("AGE", 0, 5)], "b")]
    m = score(pred, gold)
    assert m.recall == 0.0 and m.precision == 0.0


def test_missing_prediction_document_counts_as_misses():
    gold = [out([sp("AGE", 0, 5)], "a")]
    m = score([], gold)
    assert m.recall == 0.0 and m.support == 1


def test_micro_average_across_documents():
    gold = [out([sp("AGE", 0, 5), sp("NAME", 10, 15)], "a"), out([sp("DATE", 0, 4)], "b")]
    pred = [out([sp("AGE", 0, 5)], "a"), out([sp("DATE", 0, 4)], "b")]
    m = score(pred, gold)
    assert m.recall == pytest.approx(2 / 3)
    assert m.precision == 1.0
    assert m.support == 3


def test_per_category_support_and_f1():
    gold = [out([sp("AGE", 0, 5), sp("AGE", 10, 15), sp("NAME", 20, 25)])]
    pred = [out([sp("AGE", 0, 5), sp("NAME", 20, 25)])]
    m = score(pred, gold)
    assert m.per_category["AGE"]["support"] == 2
    assert m.per_category["AGE"]["recall"] == pytest.approx(0.5)
    assert m.per_category["NAME"]["f1"] == 1.0


# ---------------------------------------------------------------- validator
def _write(tmp_path, text, spans, doc_id=None):
    import hashlib

    did = doc_id or hashlib.sha256(text.encode()).hexdigest()[:16]
    p = tmp_path / "g.json"
    p.write_text(
        json.dumps(
            {"doc_id": did, "text": text, "spans": spans, "annotator": "D", "reviewed_by": "A"}
        )
    )
    return p


def test_validator_accepts_correct_file(tmp_path):
    t = "My son is 32 years old."
    p = _write(tmp_path, t, [{"text": "32 years old", "category": "AGE", "start": 10, "end": 22}])
    errs, warns, _ = validate_file(p)
    assert errs == [] and warns == []


def test_validator_catches_offset_typo(tmp_path):
    t = "My son is 32 years old."
    p = _write(tmp_path, t, [{"text": "32 years old", "category": "AGE", "start": 9, "end": 21}])
    errs, _, _ = validate_file(p)
    assert any("text[9:21]" in e for e in errs)


def test_validator_catches_bad_doc_id(tmp_path):
    p = _write(tmp_path, "hello", [], doc_id="0000000000000000")
    errs, _, _ = validate_file(p)
    assert any("doc_id" in e for e in errs)


def test_validator_catches_invalid_category(tmp_path):
    p = _write(
        tmp_path, "hello", [{"text": "hello", "category": "DIAGNOSIS", "start": 0, "end": 5}]
    )
    errs, _, _ = validate_file(p)
    assert any("invalid category" in e for e in errs)


def test_validator_catches_overlap_and_missing_offsets(tmp_path):
    t = "My son is 32 years old."
    spans = [
        {"text": "32 years old", "category": "AGE", "start": 10, "end": 22},
        {"text": "years old", "category": "AGE", "start": 13, "end": 22},
        {"text": "My son", "category": "RELATION", "start": None, "end": None},
    ]
    errs, _, _ = validate_file(_write(tmp_path, t, spans))
    assert any("overlapping" in e for e in errs)
    assert any("needs start and end" in e for e in errs)


def test_validator_quasi_must_be_offsetless(tmp_path):
    p = _write(tmp_path, "hello", [{"text": "x", "category": "QUASI_COMBO", "start": 0, "end": 2}])
    errs, _, _ = validate_file(p)
    assert any("QUASI_COMBO must have" in e for e in errs)
