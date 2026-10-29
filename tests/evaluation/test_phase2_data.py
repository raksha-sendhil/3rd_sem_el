"""Validation of Person D's Phase 3 data: gold facts and routing, people, corpus, attacks."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from sanitise.evaluation.attacks import score_round
from sanitise.loader import compute_doc_id
from sanitise.schemas import (
    RELATIONS,
    AttackDefinition,
    ExtractorOutput,
    GoldRouting,
    Importance,
    Person,
    PIICategory,
    RagAnswer,
)

DATA = Path(__file__).resolve().parents[2] / "data"
CASES = [f"case_{i:02d}" for i in range(1, 21)]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES)
def test_gold_facts(case):
    gold = load(DATA / "gold" / f"{case}.json")
    d = load(DATA / "gold_facts" / f"{case}.json")
    out = ExtractorOutput.model_validate({"doc_id": d["doc_id"], "facts": d["facts"]})
    assert out.doc_id == gold["doc_id"] == compute_doc_id(gold["text"])
    assert out.facts and d["annotator"] == "D"
    assert all(f.relation in RELATIONS for f in out.facts)
    assert any(f.importance is Importance.HIGH for f in out.facts)
    keys = [(f.subject, f.relation, f.value.lower()) for f in out.facts]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("case", CASES)
def test_gold_routing_matches_gold_spans(case):
    gold = load(DATA / "gold" / f"{case}.json")
    r = GoldRouting.model_validate(
        {
            k: v
            for k, v in load(DATA / "gold_rewrite" / f"{case}.json").items()
            if k in ("doc_id", "routing")
        }
    )
    assert r.doc_id == gold["doc_id"]
    assert [(x.text, x.category.value) for x in r.routing] == [
        (s["text"], s["category"]) for s in gold["spans"]
    ]
    placeholder = {"NAME", "PHONE", "EMAIL", "ID_NUMBER", "ADDRESS"}
    for x in r.routing:
        if x.category.value in placeholder:
            assert x.action.value == "placeholder"
        if x.category in (PIICategory.SYMPTOM, PIICategory.RELATION):
            assert x.action.value == "keep"


def people():
    return [Person.model_validate(p) for p in load(DATA / "corpus" / "people.json")]


def docs():
    return [load(p) for p in sorted((DATA / "corpus" / "docs").glob("*.json"))]


def test_people_table():
    ps = people()
    assert 100 <= len(ps) <= 140
    assert len({p.person_id for p in ps}) == len(ps) == len({p.name for p in ps})
    for p in ps:
        assert p.age_band == f"{p.age // 10 * 10}-{p.age // 10 * 10 + 9}"
        assert len(p.doc_ids) >= 2


def test_corpus_documents():
    ds = docs()
    ps = {p.person_id: p for p in people()}
    assert 280 <= len(ds) <= 320
    ids = [d["doc_id"] for d in ds]
    assert len(set(ids)) == len(ids)
    for d in ds:
        assert d["doc_id"] == compute_doc_id(d["text"]) and len(d["text"]) <= 2000
        assert d["doc_id"] in ps[d["person_id"]].doc_ids
    assert {i for p in ps.values() for i in p.doc_ids} == set(ids)


ATTACKS = sorted((DATA / "attacks").glob("*_p*.json"))


def test_attack_set_is_three_by_three():
    kinds = [AttackDefinition.model_validate(load(p)).kind for p in ATTACKS]
    assert len(kinds) == 9 and {k: kinds.count(k) for k in set(kinds)} == {
        "context_dump": 3,
        "stepwise": 3,
        "interception": 3,
    }


@pytest.mark.parametrize("path", ATTACKS, ids=lambda p: p.stem)
def test_attack_script(path):
    raw = load(path)
    a = AttackDefinition.model_validate(raw)
    target = next(p for p in people() if p.person_id == a.target_person_id)
    assert path.stem == a.attack_id and len(raw["expected_disclosure"]) == len(a.queries)
    if (
        a.kind == "interception"
    ):  # the observer sees the question: it must not contain the target's attributes
        for q in a.queries:
            hit = score_round(
                people(),
                target.person_id,
                [
                    RagAnswer(
                        pipeline="baseline",
                        query="",
                        answer=q,
                        doc_ids=[],
                        payload="",
                        latency_ms=0,
                    )
                ],
            )
            assert not set(hit.disclosed) & {
                "name",
                "phone",
                "age",
                "town",
                "occupation",
                "condition_0",
            }, (q, hit.disclosed)
    if a.kind == "stepwise":  # the pool really shrinks round by round, ending at the target
        pools = [
            score_round(
                people(), target.person_id, [_ans(" ".join(_value(target, f) for f in fields))]
            ).candidate_pool
            for fields in raw["expected_disclosure"]
        ]
        assert pools == sorted(pools, reverse=True) and pools[0] > pools[-1] and pools[-1] == 1


def _value(p: Person, field: str) -> str:
    return {
        "name": p.name,
        "age": f"{p.age} years old",
        "town": p.town,
        "occupation": p.occupation,
        "phone": p.phone or "",
        "condition_0": p.conditions[0],
    }[field]


def _ans(text: str) -> RagAnswer:
    return RagAnswer(
        pipeline="baseline", query="", answer=text, doc_ids=[], payload="", latency_ms=0
    )


def test_target_info_queries():
    qs = load(DATA / "attacks" / "target_info_queries.json")
    assert 240 <= len(qs) <= 260
    assert len({q["query"] for q in qs}) == len(qs)
    assert {q["kind"] for q in qs} == {"extract", "dump"}
    assert not any(re.search(r"\{|\}", q["query"]) for q in qs)
