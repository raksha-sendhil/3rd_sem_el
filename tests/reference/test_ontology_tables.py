"""Person B's generalisation tables (data/ontology/, plan 6.3) load, cover the corpus and never
repeat the detail they replace."""

import json
from pathlib import Path

import pytest

from sanitise.agents.rewriter.leakcheck import contains
from sanitise.agents.rewriter.ontology import ONTOLOGY_DIR, generalise, is_rare
from sanitise.schemas import PIICategory as C

PEOPLE = json.loads((ONTOLOGY_DIR.parent / "corpus" / "people.json").read_text(encoding="utf-8"))


def _table(name: str) -> dict:
    return json.loads(Path(ONTOLOGY_DIR, f"{name}.json").read_text(encoding="utf-8"))


def test_tables_have_the_planned_shape():
    for entry in _table("places").values():
        assert set(entry) == {"district", "state", "kind"}
    for phrase in _table("occupations").values():
        assert phrase.startswith(("a ", "an "))
    for entry in _table("conditions").values():
        assert entry["category"].startswith(("a ", "an ")) and isinstance(entry["rare"], bool)
    for name in ("places", "occupations", "conditions"):
        assert all(k == k.lower() for k in _table(name))


@pytest.mark.parametrize("person", PEOPLE, ids=lambda p: p["person_id"])
def test_every_corpus_town_job_and_condition_is_listed(person):
    assert person["town"].lower() in _table("places")
    assert person["occupation"].lower() in _table("occupations")
    assert all(c.lower() in _table("conditions") for c in person["conditions"])


def test_a_generalised_detail_never_repeats_itself():
    for town in _table("places"):
        assert not contains(generalise(town, C.LOCATION), town, C.LOCATION)
    for job in _table("occupations"):
        assert not contains(generalise(job, C.OCCUPATION), job, C.OCCUPATION)


def test_rare_conditions_are_the_ones_marked_rare():
    assert is_rare("amyloidosis") and is_rare("multiple myeloma")
    assert not is_rare("vertigo") and not is_rare("type 2 diabetes")
    assert generalise("Coonoor", C.LOCATION) == "a town in the Nilgiris district"
