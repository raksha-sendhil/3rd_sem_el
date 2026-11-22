"""The candidate-pool monitor: a population built from the store's sanitised records only."""

from types import SimpleNamespace

from sanitise.monitor import CandidatePoolMonitor, SessionMonitor, build_monitor
from sanitise.schemas import Action, PIICategory, RetrievalHit

C = PIICategory


def record(issue, place, job, age):
    shown = [
        (C.LEGAL_ISSUE, issue, Action.KEEP, None),
        (C.LOCATION, "raw place", Action.GENERALISE, place),
        (C.OCCUPATION, "raw job", Action.GENERALISE, job),
        (C.AGE, "raw age", Action.GENERALISE, age),
        (C.NAME, "Asha Rao", Action.PLACEHOLDER, "[CLIENT_1]"),
    ]
    decisions = [
        SimpleNamespace(category=c, span_text=s, action=a, replacement=r, start=0)
        for c, s, a, r in shown
    ]
    return SimpleNamespace(rewriter=SimpleNamespace(decisions=decisions))


class Store:
    def __init__(self):
        rows = [("cheque bounce", "a city in Kerala", "a trader", "in their 30s")] * 20
        rows += [("cheque bounce", "a city in Goa", "a trader", "in their 30s")] * 6
        rows += [("cheque bounce", "a city in Goa", "a doctor", "in their 50s")] * 3
        rows += [("maritime lien", "a city in Assam", "a sailor", "in their 40s")]
        self.rows = {f"d{i}": record(*row) for i, row in enumerate(rows)}

    def ids(self):
        return list(self.rows)

    def sanitised(self, doc_id):
        return self.rows[doc_id]


DOCTOR = [RetrievalHit(doc_id="d26", score=1.0)]
SAILOR = [RetrievalHit(doc_id="d29", score=1.0)]


def test_protocol():
    assert isinstance(build_monitor(Store()), SessionMonitor)


def test_a_question_alone_discloses_nothing():
    m = CandidatePoolMonitor(Store())
    v = m.check("s", "a doctor in Goa in their 50s?", DOCTOR, "The records do not say.")
    assert (v.verdict, v.disclosed) == ("allowed", [])


def test_content_alone_is_never_blocked():
    v = CandidatePoolMonitor(Store()).check("s", "q", SAILOR, "It was a maritime lien.")
    assert (v.verdict, v.candidate_pool, v.disclosed) == ("allowed", 1, ["legal_issue"])


def test_stepwise_attack_is_stopped_before_identification():
    m = CandidatePoolMonitor(Store())
    assert m.check("s", "q", DOCTOR, "A cheque bounce matter.").candidate_pool == 29
    assert m.check("s", "q", DOCTOR, "In Goa.").candidate_pool == 9
    v = m.check("s", "q", DOCTOR, "The client is a doctor.")
    assert (v.verdict, v.candidate_pool) == ("generalised", 9)
    assert v.revised_answer == "The client is a working professional."
    assert "doctor" not in v.model_dump_json()


def test_confirming_the_question_is_blocked():
    m = CandidatePoolMonitor(Store())
    assert m.check("s", "Is the client a doctor in Goa?", DOCTOR, "Yes.").verdict == "blocked"


def test_sessions_are_independent():
    m = CandidatePoolMonitor(Store())
    m.check("a", "q", DOCTOR, "In Goa.")
    assert m.check("b", "q", DOCTOR, "A cheque bounce matter.").candidate_pool == 29
