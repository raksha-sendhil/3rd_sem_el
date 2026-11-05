"""The fictional legal corpus (Person D): sizes, exact offsets, candidate pools, QA, attacks."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from sanitise.agents.rewriter.ontology import generalise
from sanitise.legal import generator
from sanitise.legal.vocab import CITIES, ISSUES, OCCUPATIONS, ontology_tables
from sanitise.loader import compute_doc_id
from sanitise.schemas import PIICategory as C

REPO = Path(__file__).resolve().parents[2]
LEGAL = REPO / "data" / "legal"


@pytest.fixture(scope="module")
def g() -> generator.Generated:
    return generator.build()


def test_sizes_match_the_spec(g: generator.Generated) -> None:
    assert len(g.clients) == 120
    assert len(g.docs) == 300
    assert len(g.nonmembers) == 120
    assert (len(g.train), len(g.heldout)) == (400, 40)
    assert set(Counter(len(c.doc_ids) for c in g.clients)) <= {2, 3}
    assert {d.template for d in g.docs} == set(generator.TEMPLATES)


def test_offsets_are_exact_and_doc_ids_are_content_hashes(g: generator.Generated) -> None:
    for d in g.docs + g.nonmembers + g.train:
        assert d.doc_id == compute_doc_id(d.text)
        for s in d.spans:
            if s["start"] is not None:
                assert d.text[s["start"] : s["end"]] == s["text"]
        assert len(d.text) <= 2000  # Settings.max_doc_chars


def test_candidate_pools_narrow_as_the_spec_describes(g: generator.Generated) -> None:
    cheque = [c for c in g.clients if c.issue.name == "cheque bounce"]
    mh = [c for c in cheque if c.city.state == "Maharashtra"]
    doctors = [c for c in mh if c.group == "doctor"]
    fifties = [c for c in doctors if 50 <= c.age < 60]
    assert (len(cheque), len(mh), len(doctors), len(fifties)) == (40, 12, 3, 1)
    assert fifties[0].client_id == "c001"
    rare = sum(c.issue.rare for c in g.clients)
    assert 0.06 <= rare / len(g.clients) <= 0.10


def test_targets_are_the_first_three_clients(g: generator.Generated) -> None:
    targets = [(c.issue.name, c.city.name, c.occupation, c.age) for c in g.clients[:3]]
    assert targets == [
        ("cheque bounce", "Pune", "cardiologist", 52),
        ("tenancy eviction", "Mysuru", "schoolteacher", 44),
        ("consumer complaint", "Hyderabad", "software engineer", 31),
    ]
    for c in g.clients[:3]:
        assert len(c.doc_ids) == 3


def test_names_and_emails_are_unique_and_reserved(g: generator.Generated) -> None:
    names = [c.name for c in g.clients + g.nonmember_clients()]
    assert len(names) == len(set(names))
    assert all(c.email.endswith(("@example.in", "@example.com")) for c in g.clients)


def test_qa_answers_are_in_the_answering_documents(g: generator.Generated) -> None:
    items = generator.qa_items(g.docs)
    assert len(items) == 40
    by_id = {d.doc_id: d.text.lower() for d in g.docs}
    for q in items:
        assert q["doc_ids"]
        for doc_id in q["doc_ids"]:
            assert all(e.lower() in by_id[doc_id] for e in q["expected"])


def test_attacks_cover_three_kinds_and_three_targets(g: generator.Generated) -> None:
    attacks = generator.attacks(g.clients)
    assert Counter(a["kind"] for a in attacks) == {"direct": 3, "linkage": 3, "stepwise": 3}
    assert all(len(a["queries"]) == 4 for a in attacks if a["kind"] == "stepwise")
    assert {a["target_person_id"] for a in attacks} == {"c001", "c002", "c003"}


def test_gold_routing_follows_the_policy_table(g: generator.Generated) -> None:
    doc = next(d for d in g.docs if d.template == "intake_note")
    routes = {r["category"]: r["action"] for r in generator.gold_routing(doc)["routing"]}
    assert routes["NAME"] == "placeholder" and routes["COURT"] == "generalise"
    assert routes["LEGAL_ISSUE"] == "keep"
    for fact in generator.gold_facts(doc)["facts"]:
        assert fact["value"] in doc.text


def test_heldout_documents_are_about_clients_never_seen_in_training(
    g: generator.Generated,
) -> None:
    train_clients = {d.client.client_id for d in g.train}
    assert not train_clients & {d.client.client_id for d in g.heldout}
    seen = "\n".join(d.text for d in g.train)
    people = {c for d in g.heldout for c in (d.client.name, d.client.opposing, d.client.email)}
    assert not [p for p in people if p in seen]


def test_committed_data_matches_the_generator(g: generator.Generated) -> None:
    """data/legal/ is the generator's output for the default seed; regenerate after edits."""
    clients = json.loads((LEGAL / "clients.json").read_text(encoding="utf-8"))
    assert [c["doc_ids"] for c in clients] == [c.doc_ids for c in g.clients]
    assert len(list((LEGAL / "corpus").glob("*.json"))) == 300


def test_relocate_spans_after_rewording() -> None:
    spans = [
        {"text": "Rohan", "category": "NAME", "start": 0, "end": 5},
        {"text": "Pune", "category": "LOCATION", "start": 15, "end": 19},
        {"text": "x + y", "category": "QUASI_COMBO", "start": None, "end": None},
    ]
    moved = generator.relocate_spans("In Pune, Rohan said.", spans)
    assert moved is not None
    assert [(s["text"], s["start"]) for s in moved] == [("Pune", 3), ("Rohan", 9), ("x + y", None)]
    assert generator.relocate_spans("Rohan said.", spans) is None


def test_ontology_tables_cover_the_vocabulary() -> None:
    tables = ontology_tables()
    assert {c.name.lower() for c in CITIES} <= set(tables["places"])
    assert set(OCCUPATIONS) == set(tables["occupations"])
    assert {i.name for i in ISSUES} == set(tables["legal_issues"])
    assert sum(v["rare"] for v in tables["legal_issues"].values()) == 4


def test_committed_ontology_drives_the_ladders() -> None:
    assert generalise("Kozhikode", C.LOCATION) == "a city in Kerala"
    assert generalise("Kozhikode", C.LOCATION, steps=2) == "a city in southern India"
    assert generalise("schoolteacher", C.OCCUPATION) == "an educator"
    assert generalise("maritime salvage claim", C.LEGAL_ISSUE) == "a specialised admiralty dispute"


def test_amounts_use_indian_formats() -> None:
    import random

    rng = random.Random(1)
    seen = {generator.format_amount(42_500_000, rng) for _ in range(40)}
    assert "₹4,25,00,000" in seen and "₹4.2 crore" in seen
    assert generator._indian_grouping(425_000) == "4,25,000"
    assert generator._indian_grouping(999) == "999"
