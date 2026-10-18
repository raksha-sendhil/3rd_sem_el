from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from sanitise.api.app import create_app
from sanitise.config import Settings
from sanitise.errors import LLMUnreachable
from sanitise.evaluation import scoring
from sanitise.llm import MockLLM
from sanitise.loader import compute_doc_id
from sanitise.schemas import DetectorOutput, DetectorResult, SpanMetrics
from tests.conftest import SAMPLE_QUASI, SAMPLE_RULE_HITS, SAMPLE_TEXT


@pytest.fixture
def llm() -> MockLLM:
    return MockLLM(handler=lambda call: SAMPLE_QUASI)


@pytest.fixture
def client(fake_rules: None, llm: MockLLM) -> Iterator[TestClient]:
    app = create_app(
        Settings(llm_backend="mock", transport_retries=0, ollama_base_url="http://127.0.0.1:9"),
        llm=llm,
    )
    with TestClient(app) as c:
        yield c


def test_detect(client: TestClient) -> None:
    response = client.post("/v1/detect", json={"text": SAMPLE_TEXT})
    assert response.status_code == 200, response.text
    result = DetectorResult.model_validate(response.json())
    assert result.doc_id == compute_doc_id(SAMPLE_TEXT)
    assert result.meta.rule_span_count == len(SAMPLE_RULE_HITS)
    assert result.meta.llm_span_count == 2


def test_detect_options(client: TestClient, llm: MockLLM) -> None:
    response = client.post(
        "/v1/detect", json={"doc_id": "case_07", "text": SAMPLE_TEXT, "options": {"llm": False}}
    )
    assert response.status_code == 200
    assert response.json()["doc_id"] == "case_07"
    assert response.json()["meta"]["model"] == "none"
    assert llm.calls == []

    response = client.post(
        "/v1/detect", json={"text": SAMPLE_TEXT, "options": {"model": "qwen2.5:7b-instruct"}}
    )
    assert response.json()["meta"]["model"] == "qwen2.5:7b-instruct"


def test_detect_rules_only_endpoint(client: TestClient, llm: MockLLM) -> None:
    response = client.post("/v1/detect/rules", json={"doc_id": "d", "text": SAMPLE_TEXT})
    assert response.status_code == 200
    output = DetectorOutput.model_validate(response.json())
    assert {s.method for s in output.spans} == {"rule"}
    assert llm.calls == []


def test_detect_llm_only_endpoint(client: TestClient) -> None:
    explicit = [{"text": "My son", "category": "RELATION", "start": 0, "end": 6, "method": "rule"}]
    response = client.post(
        "/v1/detect/llm", json={"doc_id": "d", "text": SAMPLE_TEXT, "explicit": explicit}
    )
    assert response.status_code == 200, response.text
    assert {s["method"] for s in response.json()["spans"]} == {"llm"}


def test_doc_too_long_envelope(client: TestClient) -> None:
    response = client.post("/v1/detect", json={"text": "x" * 2001})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "DOC_TOO_LONG"
    assert response.json()["error"]["stage"] == "load"


def test_schema_failure_envelope(fake_rules: None) -> None:
    app = create_app(Settings(llm_backend="mock"), llm=MockLLM(["nope"] * 10, retries=2))
    with TestClient(app) as c:
        response = c.post("/v1/detect", json={"text": SAMPLE_TEXT})
    assert response.status_code == 502
    error = response.json()["error"]
    assert (error["code"], error["stage"], error["attempts"]) == (
        "SCHEMA_VALIDATION_FAILED",
        "llm_pass",
        3,
    )
    assert len(error["raw_output_sha256"]) == 64


def test_unreachable_envelope(fake_rules: None) -> None:
    app = create_app(Settings(llm_backend="mock"), llm=MockLLM([LLMUnreachable("down")]))
    with TestClient(app) as c:
        response = c.post("/v1/detect", json={"text": SAMPLE_TEXT})
    assert response.status_code == 503 and response.json()["error"]["code"] == "LLM_UNREACHABLE"


def test_bad_request_body_is_422(client: TestClient) -> None:
    assert client.post("/v1/detect", json={"doc_id": "x"}).status_code == 422
    bad = [{"text": "x", "category": "DIAGNOSIS", "start": 0, "end": 1, "method": "rule"}]
    assert client.post("/v1/detect/llm", json={"text": "x", "explicit": bad}).status_code == 422


def test_score(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    gold = [{"doc_id": "d", "spans": []}]
    response = client.post("/v1/score", json={"pred": gold, "gold": gold})
    assert response.status_code == 501  # Person D's placeholder

    metrics = SpanMetrics(precision=1, recall=1, f1=1, support=0, per_category={})
    monkeypatch.setattr(scoring, "score", lambda pred, gold: metrics)
    response = client.post("/v1/score", json={"pred": gold, "gold": gold})
    assert response.status_code == 200 and response.json()["recall"] == 1


def test_health_with_mock_backend(client: TestClient) -> None:
    response = client.get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "status": "ok",
        "ollama_reachable": False,
        "model": "mock-llm",
        "pipeline_version": "0.1.0",
    }
