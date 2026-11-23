"""Long requests run as background jobs (POST /v1/jobs, then GET /v1/jobs/{id})."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from sanitise.api.app import create_app
from sanitise.config import Settings


def _wait(client: TestClient, job_id: str) -> dict:
    for _ in range(100):
        state = client.get(f"/v1/jobs/{job_id}").json()
        if state["status"] == "done":
            return state
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_a_job_runs_a_request_and_hands_its_answer_over_once() -> None:
    with TestClient(create_app(Settings(llm_backend="mock"))) as client:
        job = client.post("/v1/jobs", json={"method": "POST", "path": "/v1/detect",
                                            "body": {"text": "Asha Rao, 54, of Pune."}}).json()  # fmt: skip
        state = _wait(client, job["job_id"])
        assert state["http_status"] == 200 and state["body"]["output"]["spans"]
        assert client.get(f"/v1/jobs/{job['job_id']}").status_code == 404


def test_an_error_response_is_handed_over_and_odd_paths_are_refused() -> None:
    with TestClient(create_app(Settings(llm_backend="mock"))) as client:
        job = client.post("/v1/jobs", json={"method": "POST", "path": "/v1/ingest", "body": {}})
        assert _wait(client, job.json()["job_id"])["http_status"] >= 400
        assert client.post("/v1/jobs", json={"path": "/v1/jobs/x"}).status_code == 400
        assert client.post("/v1/jobs", json={"path": "/admin"}).status_code == 400
