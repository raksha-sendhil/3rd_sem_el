"""The candidate-pool session monitor (Phase 3b, legal build spec step 12)."""

import json

from sanitise.monitor import AllowAllMonitor, CandidatePoolMonitor, SessionMonitor, build_monitor


def client(i, **over):
    row = {
        "client_id": f"c{i:03d}",
        "name": f"Client {i}",
        "age": 40,
        "city": "Mumbai",
        "district": "Mumbai",
        "state": "Maharashtra",
        "occupation": "teacher",
        "occupation_group": "educator",
        "legal_issue": "cheque bounce",
        "court": "Mumbai District Court",
        "filing_date": "2023-05-01",
    }
    row.update(over)
    return row


def table():
    """The spec's shape: 40 cheque-bounce clients, 12 in Pune, 3 of those doctors, 1 aged 50-59,
    and 10 land disputes in Pune."""
    rows = [client(i) for i in range(28)]
    rows += [client(100 + i, city="Pune", district="Pune") for i in range(9)]
    rows += [
        client(
            200 + i,
            city="Pune",
            district="Pune",
            occupation="cardiologist",
            occupation_group="doctor",
            age=age,
            filing_date=f"{year}-03-12",
        )
        for i, (age, year) in enumerate([(52, 2024), (38, 2022), (41, 2021)])
    ]
    rows += [
        client(
            300 + i,
            city="Pune",
            district="Pune",
            legal_issue="land dispute",
            occupation="cardiologist" if i < 2 else "engineer",
            occupation_group="doctor" if i < 2 else "engineer",
            filing_date="2024-01-10" if i == 0 else "2020-06-01",
            age=55 if i == 0 else 30,
        )
        for i in range(10)
    ]
    return rows


def test_protocol():
    assert isinstance(CandidatePoolMonitor(table()), SessionMonitor)


def test_disclosure_uses_the_tables_vocabulary():
    m = CandidatePoolMonitor(table())
    got = m.disclosed_in("Cheque-bounce matters for doctors in their 50s in Pune, NI Act 1881?")
    assert got == {
        "legal_issue": {"cheque bounce"},
        "city": {"pune"},
        "district": {"pune"},
        "occupation_group": {"doctor"},
        "age_band": {"50s"},
    }  # 1881 is no client's filing year, so it constrains nothing


def test_stepwise_attack_is_blocked_before_identification():
    m = CandidatePoolMonitor(table(), min_pool=5)
    r1 = m.check(
        "s",
        "Any land disputes in Pune?",
        [],
        "Yes, several land disputes in a city in Maharashtra.",
    )
    assert r1.verdict == "allowed" and r1.candidate_pool == 10
    r2 = m.check("s", "Which one involves a cardiologist?", [], "One involved a doctor.")
    assert r2.verdict == "blocked"
    assert r2.candidate_pool == 10  # the refused round is not committed
    assert "occupation" in r2.reason
    r3 = m.check("s", "What year was it filed?", [], "It was filed in 2024.")
    assert r3.verdict == "blocked"  # land dispute + Pune + 2024 leaves one client
    assert m.pool(m.session("s")) >= 5


def test_innocent_questions_stay_allowed():
    m = CandidatePoolMonitor(table())
    r = m.check(
        "t",
        "What did we advise in cheque bounce matters?",
        [],
        "Send a legal notice within 30 days.",
    )
    assert r.verdict == "allowed" and r.candidate_pool == 40
    r = m.check("t", "And in Maharashtra?", [], "The same advice applied.")
    assert r.verdict == "allowed" and r.candidate_pool == 40
    assert r.disclosed == ["legal_issue", "state"]


def test_sessions_are_independent():
    m = CandidatePoolMonitor(table())
    m.check("a", "land disputes in Pune", [], "")
    assert m.check("b", "cardiologist", [], "").verdict == "allowed"


def test_build_monitor(tmp_path):
    assert isinstance(build_monitor(tmp_path / "missing.json"), AllowAllMonitor)
    path = tmp_path / "clients.json"
    path.write_text(json.dumps(table()), encoding="utf-8")
    assert isinstance(build_monitor(path), CandidatePoolMonitor)


def test_medical_person_table_fields_are_read():
    m = CandidatePoolMonitor([{"town": "Coonoor", "conditions": ["vertigo"], "age": 34}] * 6)
    assert m.disclosed_in("vertigo in Coonoor") == {"legal_issue": {"vertigo"}, "city": {"coonoor"}}
