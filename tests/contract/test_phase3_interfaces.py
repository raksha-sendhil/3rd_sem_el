"""Contract tests for docs/phase3_plan.md 6.2: every function keeps its signature.

They run against whatever is in the repository, placeholder or real, so a signature change
fails here on the day it happens instead of at the integration run.
"""

from __future__ import annotations

import inspect

from sanitise.agents import pipeline as offline
from sanitise.agents.extractor import canonical, importance
from sanitise.agents.extractor import pipeline as extractor_pipeline
from sanitise.agents.extractor import prompt as extractor_prompt
from sanitise.agents.rewriter import leakcheck, ontology, placeholders, routing
from sanitise.agents.rewriter import pipeline as rewriter_pipeline
from sanitise.agents.rewriter import prompt as rewriter_prompt
from sanitise.attacks import runner
from sanitise.evaluation import attacks, facts, qa, rewrite
from sanitise.llm import LLMClient, MockLLM
from sanitise.monitor import AllowAllMonitor, SessionMonitor
from sanitise.rag import answer as rag_answer
from sanitise.rag import answer_prompt
from sanitise.rag.retrieve import BM25Retriever, Retriever
from sanitise.rag.store import DocumentStore, MemoryDocumentStore, SqliteDocumentStore
from tests.contract.test_interfaces import params

SYNC = [
    (canonical.subjects, ["text", "spans"]),
    (canonical.normalise_facts, ["facts", "subjects"]),
    (importance.default_importance, ["relation"]),
    (routing.route, ["spans", "facts"]),
    (ontology.generalise, ["text", "category"]),
    (placeholders.allocate_placeholders, ["decisions"]),
    (leakcheck.leak_check, ["removed", "safe_text"]),
    (leakcheck.mask_fallback, ["safe_text", "leaks", "placeholders"]),
    (answer_prompt.build_answer_prompt, ["question", "contexts"]),
    (facts.score_facts, ["pred", "gold"]),
    (rewrite.score_rewrite, ["pred", "gold_routing", "gold_facts"]),
    (qa.score_qa, ["answers", "qa"]),
    (attacks.score_round, ["people", "target_id", "answers_so_far"]),
]

ASYNC = [
    (extractor_prompt.extract_facts, ["text", "subjects", "spans", "llm"]),
    (rewriter_prompt.reconstruct, ["facts", "decisions", "placeholders", "llm", "feedback"]),
    (extractor_pipeline.extract, ["doc_id", "text", "detected", "llm"]),
    (rewriter_pipeline.rewrite, ["doc_id", "text", "detected", "extracted", "llm"]),
    (offline.sanitise, ["doc_id", "text", "llm"]),
    (rag_answer.answer, ["ctx", "query", "pipeline", "session_id", "llm", "k"]),
    (rag_answer.compare, ["ctx", "query", "session_id", "llm", "k"]),
    (runner.run_attack, ["ctx", "attack", "people", "llm", "session_id"]),
]


def test_sync_signatures() -> None:
    for fn, expected in SYNC:
        assert params(fn) == expected, fn.__qualname__
        assert not inspect.iscoroutinefunction(fn), fn.__qualname__


def test_async_signatures() -> None:
    for fn, expected in ASYNC:
        assert params(fn) == expected, fn.__qualname__
        assert inspect.iscoroutinefunction(fn), fn.__qualname__


def test_protocols_are_satisfied(tmp_path) -> None:
    assert isinstance(MockLLM(), LLMClient)
    assert isinstance(MemoryDocumentStore(), DocumentStore)
    store = SqliteDocumentStore(tmp_path / "s.sqlite")
    assert isinstance(store, DocumentStore)
    store.close()
    assert isinstance(BM25Retriever({}), Retriever)
    assert isinstance(AllowAllMonitor(), SessionMonitor)
