"""Agent 2's deterministic logic: subjects, relation synonyms, importance (Person A)."""

from sanitise.agents.extractor.canonical import canonical_relation, normalise_facts, subjects
from sanitise.agents.extractor.importance import default_importance
from sanitise.schemas import RELATIONS, FactTuple, Importance, SensitiveSpan
from sanitise.schemas import PIICategory as C


def name(text, start, role=None):
    rationale = "test:rule" + (f" role={role}" if role else "")
    return SensitiveSpan(
        text=text,
        category=C.NAME,
        start=start,
        end=start + len(text),
        method="rule",
        rationale=rationale,
    )


def test_subjects_from_party_roles():
    spans = [
        name("Rohan Mehta", 0, "client"),
        name("Sunita Kale", 20, "party"),
        name("Mehta", 40),
        name("Priya Nair", 60),
        SensitiveSpan(
            text="my wife", category=C.RELATION, start=80, end=87, method="rule", rationale="r"
        ),
    ]
    got = {s.id: s.mentions for s in subjects("", spans)}
    assert got == {
        "writer": [],
        "client": ["Rohan Mehta", "Mehta"],
        "opposing_party": ["Sunita Kale"],
        "person_1": ["Priya Nair"],
        "writer.wife": ["my wife"],
    }


def test_relation_synonyms_land_in_the_vocabulary():
    assert canonical_relation("Advice") == "advised"
    assert canonical_relation("legal issue") == "has_legal_issue"
    assert canonical_relation("court") == "filed_in"
    assert canonical_relation("resides-in") == "lives_in"
    assert canonical_relation("filed_on") == "filed_on"
    assert canonical_relation("favourite colour") == "other"
    assert set(canonical_relation(r) for r in RELATIONS) == set(RELATIONS)


def test_normalise_merges_duplicates():
    facts = [
        FactTuple(
            subject="client", relation="advice", value="Send a legal  notice.", importance="high"
        ),
        FactTuple(
            subject="client", relation="advised", value="send a legal notice", importance="high"
        ),
        FactTuple(subject="client", relation="court", value="a High Court", importance="high"),
    ]
    out = normalise_facts(facts, [])
    assert [(f.relation, f.value) for f in out] == [
        ("advised", "Send a legal notice"),
        ("filed_in", "a High Court"),
    ]


def test_legal_importance():
    for relation in ("advised", "has_legal_issue", "filed_in", "outcome", "documents_requested"):
        assert default_importance(relation) is Importance.HIGH
    assert default_importance("opposing_party") is Importance.LOW
