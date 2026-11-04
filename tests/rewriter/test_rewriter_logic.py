"""Agent 3's deterministic logic: routing, ladders, placeholders, leak check (Person A)."""

import pytest

from sanitise.agents.rewriter.leakcheck import leak_check, mask_fallback
from sanitise.agents.rewriter.ontology import generalise, rupees
from sanitise.agents.rewriter.placeholders import allocate_placeholders
from sanitise.agents.rewriter.routing import route
from sanitise.schemas import Action, RoutingDecision, SensitiveSpan
from sanitise.schemas import PIICategory as C


def span(text, category, start=0, rationale="test:rule", method="rule"):
    if method == "llm":
        return SensitiveSpan(text=text, category=category, method="llm", rationale=rationale)
    return SensitiveSpan(
        text=text,
        category=category,
        start=start,
        end=start + len(text),
        method=method,
        rationale=rationale,
    )


def decision(text, category, action, replacement=None, reason="test"):
    return RoutingDecision(
        span_text=text,
        category=category,
        action=action,
        replacement=replacement,
        reason=reason,
        start=0,
        end=len(text),
    )


# ------------------------------------------------------------------ ladders


@pytest.mark.parametrize(
    ("text", "category", "one", "two"),
    [
        ("52-year-old", C.AGE, "in their 50s", "middle-aged"),
        ("12 March 2024", C.DATE, "March 2024", "2024"),
        ("12/03/2024", C.DATE, "March 2024", "2024"),
        ("March 2024", C.DATE, "2024", "the mid 2020s"),
        ("Pune", C.LOCATION, "a city in Maharashtra", "a city in western India"),
        ("cardiologist", C.OCCUPATION, "a doctor", "a professional"),
        ("Mehta Builders Pvt Ltd", C.ORGANIZATION, "a real-estate developer", "a company"),
        ("Bombay High Court", C.COURT, "a High Court", "a court"),
        ("Pune District Court", C.COURT, "a district court", "a court"),
        ("Rs 4,25,00,000", C.MONEY, "between ₹1 crore and ₹10 crore", "a large sum"),
        ("₹4,25,000", C.MONEY, "between ₹1 lakh and ₹10 lakh", "a modest sum"),
    ],
)
def test_ladders(text, category, one, two):
    assert generalise(text, category) == one
    assert generalise(text, category, steps=2) == two


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("2024", C.DATE),
        ("Supreme Court", C.COURT),
        ("Maharashtra", C.LOCATION),
        ("Pune", C.LOCATION),
        ("Rs 50,000", C.MONEY),
    ],
)
def test_generalised_value_never_contains_the_original(text, category):
    for steps in (1, 2):
        assert text.lower() not in generalise(text, category, steps=steps).lower()


def test_rupees():
    assert rupees("Rs 4,25,00,000") == 42_500_000
    assert rupees("₹1.2 crore") == 12_000_000
    assert rupees("25 lakh rupees") == 2_500_000
    assert rupees("no amount") is None


# ------------------------------------------------------------------ routing


def test_route_policy():
    spans = [
        span("Rohan Mehta", C.NAME, rationale="x role=client"),
        span("CS(COMM) 412/2024", C.CASE_NUMBER),
        span("HDFC0001234", C.FINANCIAL),
        span("Project Falcon", C.MATTER_CODENAME),
        span("Bombay High Court", C.COURT),
        span("Rs 4,25,000", C.MONEY),
        span("Mehta Builders Pvt Ltd", C.ORGANIZATION),
        span("cheque bounce", C.LEGAL_ISSUE),
        span("my client's wife", C.RELATION),
    ]
    got = [(d.category, d.action) for d in route(spans, [])]
    assert got == [
        (C.NAME, Action.PLACEHOLDER),
        (C.CASE_NUMBER, Action.PLACEHOLDER),
        (C.FINANCIAL, Action.PLACEHOLDER),
        (C.MATTER_CODENAME, Action.PLACEHOLDER),
        (C.COURT, Action.GENERALISE),
        (C.MONEY, Action.GENERALISE),
        (C.ORGANIZATION, Action.GENERALISE),
        (C.LEGAL_ISSUE, Action.KEEP),
        (C.RELATION, Action.KEEP),
    ]
    assert "role=client" in route(spans, [])[0].reason


def test_rare_legal_issue_is_generalised(monkeypatch):
    from sanitise.agents.rewriter import ontology

    monkeypatch.setattr(
        ontology,
        "table",
        lambda name: (
            {
                "maritime salvage claim": {
                    "category": "a specialised admiralty dispute",
                    "rare": True,
                }
            }
            if name == "legal_issues"
            else {}
        ),
    )
    [d] = route([span("maritime salvage claim", C.LEGAL_ISSUE)], [])
    assert d.action is Action.GENERALISE
    assert generalise("maritime salvage claim", C.LEGAL_ISSUE) == "a specialised admiralty dispute"


def test_quasi_combo_members_climb_two_steps():
    spans = [
        span("cardiologist", C.OCCUPATION, 0),
        span("Pune", C.LOCATION, 20),
        span("land dispute", C.LEGAL_ISSUE, 30),
        span("12 March 2024", C.DATE, 50),
        span("Rohan Mehta", C.NAME, 70),
        span("cardiologist + Pune + land dispute + 2024", C.QUASI_COMBO, method="llm"),
    ]
    ds = route(spans, [])
    assert ds[0].replacement == "a professional"
    assert ds[1].replacement == "a city in western India"
    assert ds[2].action is Action.KEEP  # something else already breaks the combination
    assert ds[3].replacement == "2024"  # "2024" is part of "12 March 2024"
    assert ds[4].replacement is None  # not a member
    assert len(ds) == len(spans)
    assert all(d.start == s.start for d, s in zip(ds, spans, strict=True))


def test_quasi_combo_of_kept_members_breaks_one():
    spans = [
        span("land dispute", C.LEGAL_ISSUE, 0),
        span("my client's wife", C.RELATION, 20),
        span("land dispute + my client's wife", C.QUASI_COMBO, method="llm"),
    ]
    ds = route(spans, [])
    assert ds[0].action is Action.GENERALISE
    assert ds[1].action is Action.KEEP


# ------------------------------------------------------------------ placeholders


def test_placeholders_by_role_and_shared_surname():
    ds = [
        decision("Rohan Mehta", C.NAME, Action.PLACEHOLDER, reason="x role=client"),
        decision("Sunita Kale", C.NAME, Action.PLACEHOLDER, reason="x role=party"),
        decision("Anil Joshi", C.NAME, Action.PLACEHOLDER, reason="x role=witness"),
        decision("Mehta", C.NAME, Action.PLACEHOLDER),
        decision("Priya Nair", C.NAME, Action.PLACEHOLDER),
        decision("CS(COMM) 412/2024", C.CASE_NUMBER, Action.PLACEHOLDER),
        decision("HDFC0001234", C.FINANCIAL, Action.PLACEHOLDER),
        decision("Project Falcon", C.MATTER_CODENAME, Action.PLACEHOLDER),
        decision("rohan  mehta", C.NAME, Action.PLACEHOLDER),
        decision("Pune", C.LOCATION, Action.GENERALISE),
    ]
    tokens = allocate_placeholders(ds)
    assert [d.replacement for d in ds] == [
        "[CLIENT_1]",
        "[PARTY_1]",
        "[WITNESS_1]",
        "[CLIENT_1]",
        "[PERSON_1]",
        "[CASE_1]",
        "[ACCOUNT_1]",
        "[MATTER_1]",
        "[CLIENT_1]",
        None,
    ]
    assert tokens["[CLIENT_1]"] == "Rohan Mehta"


def test_ambiguous_first_name_gets_its_own_token():
    ds = [
        decision("Rohan Mehta", C.NAME, Action.PLACEHOLDER),
        decision("Rohan Kale", C.NAME, Action.PLACEHOLDER),
        decision("Rohan", C.NAME, Action.PLACEHOLDER),
    ]
    allocate_placeholders(ds)
    assert [d.replacement for d in ds] == ["[PERSON_1]", "[PERSON_2]", "[PERSON_3]"]


# ------------------------------------------------------------------ leak check


REMOVED = [
    decision("Rohan Mehta", C.NAME, Action.PLACEHOLDER, "[CLIENT_1]"),
    decision("+91 98450 12345", C.PHONE, Action.PLACEHOLDER, "[PHONE_1]"),
    decision("ABCPM1234K", C.ID_NUMBER, Action.PLACEHOLDER, "[ID_1]"),
    decision("O.S. No. 412/2024", C.CASE_NUMBER, Action.PLACEHOLDER, "[CASE_1]"),
    decision("Pune", C.LOCATION, Action.GENERALISE, "a city in Maharashtra"),
    decision(
        "Mehta Builders Pvt Ltd", C.ORGANIZATION, Action.GENERALISE, "a real-estate developer"
    ),
    decision("12 March 2024", C.DATE, Action.GENERALISE, "March 2024"),
]


@pytest.mark.parametrize(
    ("safe", "leaked"),
    [
        ("[CLIENT_1] met ROHAN  MEHTA.", "ROHAN  MEHTA"),
        ("Call 9845012345 now.", "9845012345"),
        ("Call 098450-12345 now.", "098450-12345"),
        ("PAN abcpm 1234 k is on file.", "abcpm 1234 k"),
        ("The suit O.S. No 412 / 2024 was filed.", "O.S. No 412 / 2024"),
        ("Mr. Mehta agreed.", "Mehta"),
        ("Rohan Mheta agreed.", "Rohan Mheta"),
        ("The flat is in Pune.", "Pune"),
        ("Mehta Builders refused.", "Mehta"),  # "Builders" alone identifies nobody
    ],
)
def test_leak_check_catches_variants(safe, leaked):
    leaks = leak_check(REMOVED, safe)
    assert leaked in [lk.text for lk in leaks]
    for lk in leaks:
        assert safe[lk.start : lk.end] == lk.text


def test_leak_check_clean_text():
    safe = (
        "[CLIENT_1], a doctor in a city in Maharashtra, sued a real-estate developer in March "
        "2024 under [CASE_1]; reach them at [PHONE_1]. The builders delayed the flat."
    )
    assert leak_check(REMOVED, safe) == []


def test_mask_fallback_uses_each_decisions_replacement():
    safe = "Rohan Mheta in Pune called 9845012345."
    leaks = leak_check(REMOVED, safe)
    masked = mask_fallback(safe, leaks, {"[CLIENT_1]": "Rohan Mehta"})
    assert masked == "[CLIENT_1] in a city in Maharashtra called [PHONE_1]."
    assert leak_check(REMOVED, masked) == []
