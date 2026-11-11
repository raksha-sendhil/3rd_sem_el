"""The legal pipeline (src/sanitise_legal) served under /legal next to healthcare (mock mode)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sanitise.api.app import create_app
from sanitise.config import Settings

pytest.importorskip("sanitise_legal")


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(Settings(llm_backend="mock"))) as c:
        yield c


def test_healthcare_paths_are_unchanged(client: TestClient) -> None:
    docs = client.get("/v1/documents").json()
    legal_docs = client.get("/legal/v1/documents?domain=legal").json()
    assert docs and legal_docs
    # different corpora: the fixtures for healthcare, the fictional clients for legal
    assert {d["doc_id"] for d in docs}.isdisjoint({d["doc_id"] for d in legal_docs})


def test_legal_answers_come_from_the_legal_pipeline(client: TestClient) -> None:
    session = client.post("/legal/v1/sessions?domain=legal", json={}).json()["session_id"]
    body = client.post(
        "/legal/v1/compare?domain=legal",
        json={"query": "Who has a cheque bounce matter in Pune?", "session_id": session},
    ).json()
    assert body["baseline"]["answer"] and body["protected"]["answer"]
    attacks = {a["attack_id"] for a in client.get("/legal/v1/attacks?domain=legal").json()}
    assert {"direct_c001", "stepwise_c001"} <= attacks
    ledger = client.get(f"/legal/v1/sessions/{session}/monitor").json()
    assert ledger["monitor"] == "pool" and ledger["k"] == 5


def test_files_for_the_legal_agents_are_read_by_step_0(client: TestClient) -> None:
    text = "Intake note. Client Ravi Kumar, phone 98450 12345, a cheque bounce matter."
    result = client.post("/legal/v1/ingest", json={"filename": "note.txt", "text": text}).json()
    assert result["parts"] == [text]
    detected = client.post("/legal/v1/detect?domain=legal", json={"doc_id": "n", "text": text})
    assert detected.status_code == 200


def test_legal_can_be_switched_off() -> None:
    app = create_app(Settings(llm_backend="mock", legal_enabled=False))
    with TestClient(app) as c:
        assert c.get("/legal/v1/health").status_code == 404
        assert c.get("/v1/health").status_code == 200
