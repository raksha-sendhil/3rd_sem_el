"""The session monitor through the API: verdicts on /v1/query, the ledger endpoint, no leaks."""

from __future__ import annotations

from fastapi.testclient import TestClient

from sanitise.api.app import create_app
from sanitise.config import Settings
from sanitise.llm.mock import TEXT_CALL, MockLLM
from sanitise.monitor import DisclosureLedger, LedgerMonitor
from sanitise.rag.answer import RagContext
from sanitise.rag.retrieve import BM25Retriever
from sanitise.rag.store import MemoryDocumentStore
from tests.unit.test_monitor import PATIENTS, make_record

ANSWERS = [
    "Yes, one patient has vertigo.",
    "The patient with vertigo is in their 30s.",
    "They are a healthcare worker in their 30s with vertigo.",
    "They live in the Nilgiris district.",
]


def client_for(answers: list[str], monitor: bool = True) -> TestClient:
    store = MemoryDocumentStore()
    for doc_id, values in PATIENTS.items():
        store.put(doc_id, f"neurosurgeon patient {doc_id}", make_record(doc_id, *values))
    # every question retrieves p1 first (its original is the only one with "Coonoor")
    store.put("p1", "neurosurgeon Coonoor vertigo", make_record("p1", *PATIENTS["p1"]))
    ctx = RagContext(store=store, retriever=BM25Retriever({d: store.original(d) for d in PATIENTS}))
    if monitor:
        ctx.monitor = LedgerMonitor(store, ledger=DisclosureLedger(salt="t"))
    llm = MockLLM(by_schema={TEXT_CALL: list(answers)})
    return TestClient(create_app(Settings(llm_backend="mock"), llm=llm, rag=ctx))


def test_stepwise_questions_get_allowed_then_blocked() -> None:
    with client_for(ANSWERS) as client:
        session = client.post("/v1/sessions").json()["session_id"]
        verdicts = []
        for question in ["vertigo Coonoor?", "Coonoor age?", "Coonoor job?", "Coonoor town?"]:
            r = client.post(
                "/v1/query",
                json={"query": question, "pipeline": "protected", "session_id": session, "k": 1},
            )
            assert r.status_code == 200, r.text
            body = r.json()
            # neither the draft nor the generalised text travels as a field of its own
            assert "draft_answer" not in body and "revised_answer" not in body["monitor"]
            verdicts.append((body["monitor"]["verdict"], body["monitor"]["candidate_pool"]))
        assert verdicts[:3] == [("allowed", 11), ("allowed", 8), ("allowed", 6)]
        assert verdicts[3][0] == "blocked"
        assert "cannot be answered" in body["answer"] and "Nilgiris" not in body["answer"]

        ledger = client.get(f"/v1/sessions/{session}/monitor").json()
        assert (ledger["monitor"], ledger["k"], ledger["max_attributes"]) == ("ledger", 5, 3)
        assert [(e["attribute_type"], e["round"]) for e in ledger["entries"]] == [
            ("condition", 1),
            ("age", 2),
            ("occupation", 3),
        ]
        assert "vertigo" not in str(ledger)  # types and rounds only, never values
        assert client.get("/v1/sessions/nope/monitor").status_code == 404


def test_generalised_answer_replaces_the_draft() -> None:
    with client_for(["They live in the Nilgiris district."]) as client:
        session = client.post("/v1/sessions").json()["session_id"]
        body = client.post(
            "/v1/query",
            json={"query": "Coonoor town?", "pipeline": "protected", "session_id": session, "k": 1},
        ).json()
        assert body["monitor"]["verdict"] == "generalised"
        assert body["answer"] == "They live in a town in Tamil Nadu."


def test_monitor_off_allows_everything() -> None:
    with client_for(ANSWERS[3:], monitor=False) as client:
        session = client.post("/v1/sessions").json()["session_id"]
        body = client.post(
            "/v1/query",
            json={"query": "Coonoor town?", "pipeline": "protected", "session_id": session, "k": 1},
        ).json()
        assert body["monitor"]["verdict"] == "allowed" and "Nilgiris" in body["answer"]
        assert client.get(f"/v1/sessions/{session}/monitor").json()["monitor"] == "off"
