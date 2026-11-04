"""Domains: which prompts, data and models the pipeline uses for a request.

The pipeline is the same in every domain (Agent 1 rules + LLM pass + merge, Agent 2 facts,
Agent 3 routing + rewrite + leak check, search over originals, answer, session monitor). A
domain changes only the context: the prompts the agents get, the corpus and client table, and
which models run. Each domain is a ``DomainProfile``:

* built-in defaults below (data paths for ``medical`` and ``legal``);
* overridden and extended by ``config/domains.json`` (``SANITISE_DOMAINS_FILE``), which is
  where the team picks the models per domain and adds the third domain.

A profile field left ``null`` falls back to the global setting (``SANITISE_MODEL`` and so on),
so the medical set-up behaves exactly as before. ``settings_for(settings, domain)`` returns a
``Settings`` copy with the profile applied; every model and path builder takes that copy.

The domain of the current request comes from ``use_domain`` (the API sets it from
``?domain=`` or the ``X-Sanitise-Domain`` header), else ``SANITISE_DOMAIN``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from sanitise.config import Settings, get_settings

log = logging.getLogger(__name__)

Domain = str  # a key of profiles(); "medical" and "legal" are built in

REPO_ROOT = Path(__file__).resolve().parents[2]


class DomainProfile(BaseModel):
    name: str
    label: str = ""
    description: str = ""
    # which prompt family the agents use ("medical" or "legal"; a new domain reuses one until
    # its own prompts are written)
    prompts: str = "medical"
    # models: null = the global setting
    model: str | None = None  # Agents 2-3 (and Agent 1's LLM pass unless agent1_backend says)
    answer_backend: str | None = None  # same | gemini | ollama | mock
    gemini_model: str | None = None
    agent1_backend: str | None = None  # same | transformers
    agent1_base_model: str | None = None
    agent1_adapter: str | None = None
    # data
    store_path: str | None = None
    attacks_dir: str | None = None
    people_path: str | None = None
    clients_path: str | None = None
    index_dir: str | None = None
    retrieval_k: int | None = None
    mock_corpus: str | None = None  # documents mock mode serves for this domain


_BUILT_IN: dict[str, dict[str, Any]] = {
    "medical": {
        "label": "Medical",
        "description": "Patients' messages to doctors and workplace emails (Phase 2-3).",
        "prompts": "medical",
        "store_path": "data/store.sqlite",
        "attacks_dir": "data/attacks",
        "people_path": "data/corpus/people.json",
        "index_dir": "data/index",
        "retrieval_k": 3,
        "mock_corpus": "data/fixtures",
    },
    "legal": {
        "label": "Legal",
        "description": "A litigation firm's past matters (legal build spec).",
        "prompts": "legal",
        "store_path": "data/legal/store.sqlite",
        "attacks_dir": "data/legal/attacks",
        "people_path": "data/legal/clients.json",
        "clients_path": "data/legal/clients.json",
        "index_dir": "data/legal/index",
        "retrieval_k": 4,
        "mock_corpus": "data/legal/demo/docs",
    },
}

_SETTING_FIELDS = (
    "model",
    "answer_backend",
    "gemini_model",
    "agent1_backend",
    "agent1_base_model",
    "agent1_adapter",
    "store_path",
    "attacks_dir",
    "people_path",
    "clients_path",
    "index_dir",
    "retrieval_k",
)


@lru_cache(maxsize=4)
def _load(path: str) -> dict[str, DomainProfile]:
    merged = {name: dict(values) for name, values in _BUILT_IN.items()}
    file = Path(path)
    if not file.is_absolute() and not file.is_file():
        file = REPO_ROOT / path
    if file.is_file():
        for name, values in json.loads(file.read_text(encoding="utf-8")).items():
            if name.startswith("_"):  # "_comment" keys
                continue
            merged.setdefault(name, {}).update(values)
    return {
        name: DomainProfile(name=name, **{k: v for k, v in values.items() if k != "name"})
        for name, values in merged.items()
    }


def profiles(settings: Settings | None = None) -> dict[str, DomainProfile]:
    settings = settings or get_settings()
    return _load(settings.domains_file)


def profile(domain: str | None = None, settings: Settings | None = None) -> DomainProfile:
    known = profiles(settings)
    name = domain or current_domain()
    if name not in known:
        raise ValueError(f"unknown domain {name!r}; choose from {sorted(known)}")
    return known[name]


def settings_for(settings: Settings, domain: str | None = None) -> Settings:
    """``settings`` with the domain profile's models and paths applied."""
    p = profile(domain, settings)
    update = {f: getattr(p, f) for f in _SETTING_FIELDS if getattr(p, f) is not None}
    return settings.model_copy(update={**update, "domain": p.name})


_override: ContextVar[str | None] = ContextVar("sanitise_domain", default=None)


def current_domain() -> Domain:
    return _override.get() or get_settings().domain


def prompt_family() -> str:
    """ "medical" or "legal": which prompts the agents use in the current domain."""
    name = current_domain()
    known = profiles()
    return known[name].prompts if name in known else name


def is_legal() -> bool:
    return prompt_family() == "legal"


@contextmanager
def use_domain(domain: Domain) -> Iterator[None]:
    token = _override.set(domain)
    try:
        yield
    finally:
        _override.reset(token)
