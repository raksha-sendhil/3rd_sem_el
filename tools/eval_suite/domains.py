"""What differs between the health and the legal run: the code that sanitises, the person table,
the answer model and the session monitor.

* **health** runs on ``main`` (the healthcare pipeline): ``sanitise build``, Agents 1-3 on the
  Gemini API, answers from the Ollama model, the ledger session monitor, person table
  ``data/corpus/people.json``.
* **legal** runs on the legal branch (``law-pipeline-final``): ``python -m sanitise.legal build``
  with the legal profile of ``config/domains.json`` (legal prompts and models), the client table
  ``data/legal/clients.json`` and that branch's candidate-pool monitor.

Everything else (the questions, the answer loop, the metrics) is shared and only uses what both
branches have: ``sanitise.rag.answer.answer``, ``RagContext``, ``SqliteDocumentStore``,
``build_retriever`` and ``sanitise.evaluation.attacks.score_round``.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Domain:
    name: str
    repo: Path  # the clone whose code runs (main for health, the legal branch or main for legal)
    people_path: Path
    corpus_dir: Path  # the generated corpus: one JSON file per document
    noun: str  # "patient" / "client"
    k: int  # documents retrieved per question (each pipeline's own setting)
    # the package that runs it: "sanitise" (main's healthcare code, and the legal branch itself),
    # or "sanitise_legal" (the legal pipeline once merged into main by tools/import_legal.py)
    pkg: str = "sanitise"

    def mod(self, name: str):
        """A module of this domain's package, e.g. ``mod("rag.answer")``."""
        return importlib.import_module(f"{self.pkg}.{name}")

    # ------------------------------------------------------------ people and scoring

    def people(self):
        return self.mod("attacks.runner").load_people(self.people_path)

    def exact_fields(self, person) -> dict[str, str]:
        """The person's secret (exact) fields, as the pipeline's own scorer defines them."""
        attacks = self.mod("evaluation.attacks")
        fn = getattr(attacks, "_exact_values", None) or attacks._secrets
        return {k: v for k, v in fn(person).items() if v}

    def disclosed_exact(self, person, text: str) -> set[str]:
        """The person's exact fields that ``text`` states, by the pipeline's own matching rules."""
        attacks = self.mod("evaluation.attacks")
        if hasattr(attacks, "_found"):  # healthcare
            low = text.lower()
            return {f for f, v in self.exact_fields(person).items() if attacks._found(f, v, low)}
        said = attacks._norm(text)  # legal
        return {k for k, v in self.exact_fields(person).items()
                if attacks._said(said, text, k, v, person)}  # fmt: skip

    def direct_fields(self) -> tuple[str, ...]:
        return tuple(self.mod("evaluation.attacks").DIRECT)

    def matter(self, person) -> str:
        return person.conditions[0] if person.conditions else ""

    def with_matter(self, person) -> str:
        m = self.matter(person)
        return f"with {m}" if self.name == "health" else f"with a {m} matter"

    # ------------------------------------------------------------ running

    def context(self):
        if self.name == "legal":
            return self.mod("domain").use_domain("legal")
        return contextlib.nullcontext()

    def build(self, data_dir: Path, store: Path) -> None:
        """Agents 1-3 over ``data_dir/corpus/*.json`` into ``store`` (resumes)."""
        if self.name == "health":
            cmd = [sys.executable, "-m", "sanitise", "build", "--input", str(data_dir / "corpus"),
                   "--store", str(store), "--no-index"]  # fmt: skip
        else:
            cmd = [sys.executable, "-m", f"{self.pkg}.legal", "build", "--data", str(data_dir),
                   "--store", str(store), "--out", str(data_dir / "results")]  # fmt: skip
        if os.environ.get("EVAL_MOCK"):  # no model: the rule pass and the canned rewrite
            cmd.append("--mock")
        env = {**os.environ, "PYTHONPATH": str(self.repo / "src")}
        subprocess.run(cmd, cwd=self.repo, env=env, check=False)  # failed documents retry next run

    def answer_llm(self):
        if os.environ.get("EVAL_MOCK"):
            return self.mod("mock_mode").mock_llm(), "mock (no model)"
        factory = self.mod("llm.factory")
        if self.name == "health":
            settings = self.mod("config").get_settings()
            return factory.build_llm_client(
                settings, backend="ollama"
            ), f"{settings.model} (Ollama)"
        settings = self.mod("legal.__main__").legal_settings()
        llm = factory.build_answer_llm(settings, factory.build_llm_client(settings))
        label = settings.gemini_model if settings.answer_backend == "gemini" else settings.model
        return llm, f"{label} ({settings.answer_backend})"

    def monitor(self, store):
        build_monitor = self.mod("monitor").build_monitor
        if self.name == "health":
            return build_monitor(self.mod("config").get_settings(), store)
        return build_monitor(self.people_path, 5)


def get_domain(name: str, repo: Path) -> Domain:
    if name == "health":
        return Domain("health", repo, repo / "data/corpus/people.json", repo / "data/corpus/docs",
                      "patient", 3)  # fmt: skip
    if name == "legal":
        pkg = "sanitise_legal" if (repo / "src" / "sanitise_legal").is_dir() else "sanitise"
        return Domain("legal", repo, repo / "data/legal/clients.json", repo / "data/legal/corpus",
                      "client", 4, pkg)  # fmt: skip
    raise SystemExit(f"unknown domain {name!r}: health or legal")
