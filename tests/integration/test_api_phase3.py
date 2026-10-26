"""Phase 3a endpoints (docs/phase3_plan.md 6.4) in mock mode, as the dashboard calls them."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from sanitise.api.app import create_app
from sanitise.config import Settings
from tests.conftest import SAMPLE_TEXT


@pytest.fixture
def client(fake_rules: None) -> Iterator[TestClient]:
    with TestClient(create_app(Settings(llm_backend="mock"))) as c:
        yield c


def test_documents_and_viewer(client: TestClient) -> None:
    docs = client.get("/v1/documents").json()
    assert len(docs) == 20 and {"doc_id", "preview", "n_changes"} <= set(docs[0])
    view = client.get(f"/v1/documents/{docs[0]['doc_id']}").json()
    assert view["original"] and view["record"]["rewriter"]["safe_text"]
    missing = client.get("/v1/documents/nope")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "DOC_NOT_FOUND"


def test_sessions_compare_and_query(client: TestClient) -> None:
    session = client.post("/v1/sessions").json()["session_id"]
    both = client.post("/v1/compare", json={"query": "vertigo surgeon", "session_id": session})
    assert both.status_code == 200
    body = both.json()
    assert body["baseline"]["pipeline"] == "baseline" and body["protected"]["payload"]
    one = client.post(
        "/v1/query", json={"query": "kabaddi", "pipeline": "protected", "session_id": session}
    )
    assert one.status_code == 200 and one.json()["monitor"]["verdict"] == "allowed"
    trace = client.get(f"/v1/sessions/{session}").json()
    assert len(trace["rounds"]) == 1  # compare rounds are traced; single queries are not
    assert client.get("/v1/sessions/nope").json()["error"]["code"] == "SESSION_NOT_FOUND"


def test_attacks(client: TestClient) -> None:
    attacks = client.get("/v1/attacks").json()
    assert {a["kind"] for a in attacks} == {"direct", "linkage", "stepwise"}
    run = client.post("/v1/attacks/stepwise_s02/run", json={}).json()
    assert len(run["rounds"]) == 4 and run["rounds"][0]["baseline_leakage"] is not None
    assert client.post("/v1/attacks/nope/run").json()["error"]["code"] == "ATTACK_NOT_FOUND"


def test_sanitise_endpoint(client: TestClient) -> None:
    record = client.post("/v1/sanitise", json={"text": SAMPLE_TEXT}).json()
    assert "07700 900123" not in record["rewriter"]["safe_text"]
    assert record["residual_leaks"] == []


def test_openapi_lists_every_new_path(client: TestClient) -> None:
    paths = set(client.get("/openapi.json").json()["paths"])
    assert {
        "/v1/sanitise",
        "/v1/documents",
        "/v1/documents/{doc_id}",
        "/v1/sessions",
        "/v1/sessions/{session_id}",
        "/v1/query",
        "/v1/compare",
        "/v1/attacks",
        "/v1/attacks/{attack_id}/run",
    } <= paths
