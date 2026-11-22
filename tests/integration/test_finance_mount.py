"""The finance pipeline (src/sanitise_finance) served under /finance (mock mode)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sanitise.api.app import create_app
from sanitise.config import Settings


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(Settings(llm_backend="mock"))) as c:
        yield c


def test_finance_has_its_own_documents_and_attacks(client: TestClient) -> None:
    ours = client.get("/finance/v1/documents?domain=finance").json()
    health = client.get("/v1/documents").json()
    assert ours and {d["doc_id"] for d in ours}.isdisjoint({d["doc_id"] for d in health})
    attacks = client.get("/finance/v1/attacks").json()
    assert len(attacks) == 9 and all(a["target_person_id"].startswith("b") for a in attacks)


def test_finance_answers_reads_files_and_runs_an_attack(client: TestClient) -> None:
    session = client.post("/finance/v1/sessions", json={}).json()["session_id"]
    body = {"query": "Which bank is the statement from?", "session_id": session}
    assert client.post("/finance/v1/compare", json=body).status_code == 200
    assert client.get(f"/finance/v1/sessions/{session}/monitor").status_code == 200
    text = "Statement of account. Account number 11447241261."
    read = client.post("/finance/v1/ingest", json={"filename": "s.txt", "text": text})
    assert read.status_code == 200
    detected = client.post("/finance/v1/detect", json={"doc_id": "s", "text": text})
    assert detected.status_code == 200


def test_finance_can_be_switched_off() -> None:
    with TestClient(create_app(Settings(llm_backend="mock", finance_enabled=False))) as c:
        assert c.get("/finance/v1/health").status_code == 404
